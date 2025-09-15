class RSNAAneurysmDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_dir: str,
        target_shape: Tuple[int, int, int] = (32, 384, 384),  # (D,H,W)
        label_cols: Optional[Sequence[str]] = None,           # explicit multi-label columns (order respected)
    ):
        self.df = df.reset_index(drop=True).copy()
        self.input_dir = Path(input_dir)
        self.target_shape = target_shape
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        series_instance_uid = str(row['SeriesInstanceUID'])
        series_path = self.input_dir / 'series' / series_instance_uid

        # Load volume: (D,H,W) uint8
        vol = process_dicom_series_safe(str(series_path), self.target_shape)  # (D,H,W) uint8
        vol_t = torch.from_numpy(vol)  # uint8 [D,H,W]
        vol_t = vol_t.float().div_(255.0) 

        # Labels
        label_t = row[self.label_cols].values.astype(np.float32)
        label_t = torch.from_numpy(label_t)

        # Meta
        meta = {
            'series_instance_uid': series_instance_uid,
            'series_instance_uid_path': str(series_path)
        }

        return vol_t, label_t, meta

def collate(batch):
        vols, labels, metas = zip(*batch)
        vols = torch.stack(vols, dim=0)
        labels = torch.stack(labels, dim=0)
        return vols, labels, metas
