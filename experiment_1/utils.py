import random
import numpy as np
import torch

LABEL_COLS = [
    'Other Posterior Circulation',
    'Basilar Tip',
    'Right Posterior Communicating Artery',
    'Left Posterior Communicating Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Infraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Middle Cerebral Artery',
    'Left Middle Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Anterior Cerebral Artery',
    'Anterior Communicating Artery',
    'Aneurysm Present'
]

location_list = LABEL_COLS[:-1]  # All except 'Aneurysm Present'

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow autotune for 3D volumes
    torch.backends.cudnn.benchmark = True

def find_unused_series_instance_uid_list(train_df, train_localizers_df):
    train_localizers_validate_df = train_df.melt(
        id_vars=['SeriesInstanceUID'],
        value_vars=location_list,
        var_name='location',
        value_name='label'
    )

    train_localizers_validate_df = train_localizers_validate_df[train_localizers_validate_df['label'] == 1].copy()
    train_localizers_validate_df = train_localizers_validate_df[["SeriesInstanceUID", "location"]]

    train_localizers_validate_groupby_df = train_localizers_validate_df.groupby('SeriesInstanceUID')['location'].apply(
        lambda locs: sorted(list(locs))
    ).reset_index(name='location')

    train_localizers_groupby_df = train_localizers_df.groupby('SeriesInstanceUID')['location'].apply(
        lambda locs: sorted(list(locs))
    ).reset_index(name='location')

    unused_series_instance_uid_list = []
    for curr_train_localizers_validate_groupby_index, curr_train_localizers_validate_groupby_row in train_localizers_validate_groupby_df.iterrows():
        curr_train_localizers_validate_groupby_row_series_instance_uid = curr_train_localizers_validate_groupby_row['SeriesInstanceUID']
        curr_train_localizers_validate_groupby_row_location = curr_train_localizers_validate_groupby_row['location']

        curr_train_localizers_groupby_row = train_localizers_groupby_df[train_localizers_groupby_df['SeriesInstanceUID'] == curr_train_localizers_validate_groupby_row_series_instance_uid]
        if curr_train_localizers_groupby_row.empty:
            unused_series_instance_uid_list.append(curr_train_localizers_validate_groupby_row_series_instance_uid)
            print(f"UID {curr_train_localizers_validate_groupby_row_series_instance_uid} not found in train_localizers_groupby_df")
        else:
            curr_train_localizers_groupby_row_location = curr_train_localizers_groupby_row.iloc[0]['location']
            if not np.array_equal(
                np.sort(np.unique(curr_train_localizers_groupby_row_location)),
                np.sort(np.unique(curr_train_localizers_validate_groupby_row_location))
            ):
                print(f"  Localizers: {curr_train_localizers_groupby_row_location}")
                print(f"  Train labels: {curr_train_localizers_validate_groupby_row_location}")

    return unused_series_instance_uid_list