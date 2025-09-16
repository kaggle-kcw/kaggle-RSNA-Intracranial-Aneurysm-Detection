import numpy as np
import os
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage

class DICOMPreprocessorKaggle:
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_shape = target_shape
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        series_path = Path(series_path)
        series_name = series_path.name

        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))

        datasets = []
        for filepath in dicom_files:
            ds = pydicom.dcmread(filepath, force=True)
            datasets.append(ds)

        return datasets, series_name

    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        slice_info = []

        for index, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': index,
                'instance_number': getattr(ds, 'InstanceNumber', index),
            }

            position = getattr(ds, 'ImagePositionPatient', None)
            if position is not None and len(position) >= 3:
                info['z_position'] = float(position[2])
            else:
                info['z_position'] = float(info['instance_number'])

            slice_info.append(info)

        return slice_info

    def get_windowing_params(self, ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
        """
        Get windowing parameters based on modality
        """
        modality = getattr(ds, 'Modality', 'CT')

        if modality == 'CT':
            # For CT, apply CTA (angiography) settings
            return "CT", "CT"

        elif modality == 'MR':
            # For MR, skip windowing (statistical normalization only)
            return None, None
        else:
            # Unexpected modality (safety measure), using CTA windowing
            return None, None

    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """
        Apply windowing or statistical normalization
        """
        if center is not None and width is not None:
            # Windowing processing (for CT/CTA)
            # Applied windowing: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")

            # Statistical normalization (for CT as well)
            # Normalize using 1-99 percentiles
            p1, p99 = np.percentile(img, [1, 99])
            p1, p99 = 0, 500

            if p99 > p1:
                # Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)

                return result
            else:
                # Fallback: min-max normalization
                # Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    # If image has no variation
                    return np.zeros_like(img, dtype=np.uint8)
        else:
            # Statistical normalization (for MR)
            # Normalize using 1-99 percentiles
            # Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]
            p1, p99 = np.percentile(img, [1, 99])

            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)

                return result
            else:
                # Fallback: min-max normalization
                # Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    # If image has no variation
                    return np.zeros_like(img, dtype=np.uint8)

    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Extract 2D pixel array from DICOM and apply preprocessing (for 2D DICOM series)
        """
        img = ds.pixel_array.astype(np.float32)

        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            # 3D DICOM in 2D processing - using middle frame
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]

        # Convert color image to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        return img

    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)

        if current_shape == target_shape:
            return volume

        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]

        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

        # Clip to exact size just in case
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]

        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]

        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        return resized_volume.astype(np.uint8)

    def process_series(self, series_path: str) -> np.ndarray:
        datasets, series_name = self.load_dicom_series(series_path)

        # Handle cases where no DICOM files are found in the series directory
        if not datasets:
            # Return a zero-filled volume of the target shape
            return np.zeros(self.target_shape, dtype=np.uint8)

        # Check first DICOM to determine 3D/2D
        first_ds = datasets[0]
        first_img = first_ds.pixel_array

        if len(datasets) == 1 and first_img.ndim == 3:
            # Case 1: Single 3D DICOM file
            return self._process_single_3d_dicom(first_ds)
        else:
            # Case 2: Multiple 2D DICOM files
            return self._process_multiple_2d_dicoms(datasets)


    def _process_single_3d_dicom(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Process single 3D DICOM file
        """
        volume = ds.pixel_array.astype(np.float32)
        window_center, window_width = self.get_windowing_params(ds)

        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)

        return final_volume

    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset]) -> np.ndarray:
        """
        Process multiple 2D DICOM files
        """
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'])
        processed_slices = []

        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))

            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)

        return final_volume

def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
    volume = preprocessor.process_series(series_path)
    return volume

    