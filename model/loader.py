import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Any

class SequenceDataset(TensorDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_cont: int,
        res_map: dict,
        max_seq_len: int = 128,
        columns_to_shift: Optional[List[str]] = None,
    ) -> None:
        self.res_map = res_map
        self.device = 'cpu'  # Hard-coded device; update if needed.

        # Shift specified columns if provided.
        if columns_to_shift is not None:
            for col in columns_to_shift:
                df[col] = df.groupby(['game_pk', 'pitcher'])[col].shift(1)
            first_indices = df.groupby(['game_pk', 'pitcher']).head(1).index
            df.loc[first_indices, columns_to_shift] = 0

        # Group the DataFrame by game and pitcher.
        grouped = df.groupby(['game_pk', 'pitcher'])
        batters, pitchers, catchers = [], [], []
        categorical_feats, continuous_feats = [], []
        targets, mapped_targets = [], []

        for _, group in grouped:
            batters.append(self._pad_sequence(group.iloc[:, 2].values, max_seq_len))
            pitchers.append(self._pad_sequence(group.iloc[:, 3].values, max_seq_len))
            catchers.append(self._pad_sequence(group.iloc[:, 4].values, max_seq_len))
            categorical_feats.append(self._pad_sequence(group.iloc[:, 5:-num_cont].values, max_seq_len))
            continuous_feats.append(self._pad_sequence(group.iloc[:, -num_cont:].values, max_seq_len))
            targets.append(self._pad_sequence(group.iloc[:, 0].values, max_seq_len))
            mapped_targets.append(self._pad_sequence(group.iloc[:, 0].map(res_map).values, max_seq_len))

        # Convert the lists to tensors.
        tensors = [
            torch.tensor(np.array(batters, dtype=np.int32), dtype=torch.int32, device=self.device),
            torch.tensor(np.array(pitchers, dtype=np.int32), dtype=torch.int32, device=self.device),
            torch.tensor(np.array(catchers, dtype=np.int32), dtype=torch.int32, device=self.device),
            torch.tensor(np.array(categorical_feats, dtype=np.int32), dtype=torch.int32, device=self.device),
            torch.tensor(np.array(continuous_feats, dtype=np.float32), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(targets, dtype=np.int32), dtype=torch.int32, device=self.device),
            torch.tensor(np.array(mapped_targets, dtype=np.int32), dtype=torch.int32, device=self.device),
        ]
        super().__init__(*tensors)

    def _pad_sequence(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or truncate a sequence to a fixed target length."""
        seq_len = len(seq)
        if seq_len < target_len:
            if seq.ndim == 1:
                return np.pad(seq, (0, target_len - seq_len), mode='constant')
            else:
                return np.pad(seq, ((0, target_len - seq_len), (0, 0)), mode='constant')
        elif seq_len > target_len:
            return seq[:target_len]
        return seq

def get_loaders(
    datafile: str,
    trainIDfile: str,
    valIDfile: str,
    testIDfile: str,
    res_map_file: str,
    num_cont: int,
    batch_size: int = 32,
    max_seq_len: int = 128,
    num_workers: int = 1,
    columns_to_shift: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load the main dataset.
    df = pd.read_parquet(datafile)

    # Load train, validation, and test IDs.
    with open(trainIDfile, 'rb') as f:
        train_ids = pickle.load(f)
    with open(valIDfile, 'rb') as f:
        val_ids = pickle.load(f)
    with open(testIDfile, 'rb') as f:
        test_ids = pickle.load(f)

    # Load the result mapping.
    with open(res_map_file, 'rb') as f:
        res_map = pickle.load(f)

    # Helper to extract a (game_pk, pitcher) tuple for a given row index.
    def get_group(idx: int) -> Tuple[Any, Any]:
        row = df.iloc[idx]
        return (row['game_pk'], row['pitcher'])

    # Build sets of (game_pk, pitcher) groups.
    train_groups = {get_group(idx) for idx in train_ids}
    val_groups = {get_group(idx) for idx in val_ids}
    test_groups = {get_group(idx) for idx in test_ids}

    # Filter the DataFrame based on group membership.
    idx_train = df.set_index(['game_pk', 'pitcher']).index.isin(train_groups)
    idx_val = df.set_index(['game_pk', 'pitcher']).index.isin(val_groups)
    idx_test = df.set_index(['game_pk', 'pitcher']).index.isin(test_groups)
    train_df, val_df, test_df = df[idx_train], df[idx_val], df[idx_test]

    # Create dataset objects.
    train_dataset = SequenceDataset(train_df, num_cont, res_map, max_seq_len, columns_to_shift)
    val_dataset = SequenceDataset(val_df, num_cont, res_map, max_seq_len, columns_to_shift)
    test_dataset = SequenceDataset(test_df, num_cont, res_map, max_seq_len, columns_to_shift)

    # Create DataLoader instances.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def load_inputs(
    df: pd.DataFrame,
    num_cont: int,
    pitcher_id: Any,
    seq_len: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract and pad sequences for a given pitcher.

    Returns a tuple of tensors for:
        batter_seq, pitcher_seq, catcher_seq, categorical_seq, continuous_seq, target_seq.
    """
    pitcher_data = df[df["pitcher"] == pitcher_id]
    if pitcher_data.shape[0] > seq_len:
        pitcher_data = pitcher_data.iloc[:seq_len]

    target_seq = pitcher_data.iloc[:, 0].values
    batter_seq = pitcher_data.iloc[:, 2].values
    pitcher_seq = pitcher_data.iloc[:, 3].values
    catcher_seq = pitcher_data.iloc[:, 4].values
    categorical_seq = pitcher_data.iloc[:, 5:-num_cont].values
    continuous_seq = pitcher_data.iloc[:, -num_cont:].values

    def pad_tensor(arr: np.ndarray, is_2d: bool = False) -> torch.Tensor:
        tensor = torch.tensor(arr)
        current_len = tensor.shape[0]
        pad_length = seq_len - current_len
        if pad_length > 0:
            if is_2d:
                tensor = nn.functional.pad(tensor, (0, 0, 0, pad_length), value=0)
            else:
                tensor = nn.functional.pad(tensor, (0, pad_length), value=0)
        return tensor.unsqueeze(0)

    cat_tensor = pad_tensor(categorical_seq, is_2d=True)
    cont_tensor = pad_tensor(continuous_seq, is_2d=True)
    batter_tensor = pad_tensor(batter_seq)
    pitcher_tensor = pad_tensor(pitcher_seq)
    catcher_tensor = pad_tensor(catcher_seq)
    target_tensor = pad_tensor(target_seq)

    return batter_tensor, pitcher_tensor, catcher_tensor, cat_tensor, cont_tensor, target_tensor
