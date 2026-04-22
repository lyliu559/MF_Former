import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from configs import config_tp as config


class StandardScaler:
    """Standard normalization helper."""

    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 0 else 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def normalize_data(data_path, variable_name, mean=None, std=None):
    """
    Load and normalize a WeatherBench variable.

    Parameters
    ----------
    data_path : str
        Path to the NetCDF file.
    variable_name : str
        Variable name in the NetCDF file, e.g. 'tp', 'r', 't', 'u', 'v'.
    mean : float, optional
        External mean for normalization.
    std : float, optional
        External std for normalization.

    Returns
    -------
    np.ndarray
        Normalized array with shape [time, lat, lon].
    """
    with xr.open_dataset(data_path) as data:
        var_data = data[variable_name].values

    if mean is None:
        mean = np.mean(var_data)
    if std is None:
        std = np.std(var_data)

    scaler = StandardScaler(mean, std)
    normalized_data = scaler.transform(var_data).astype(np.float32)

    assert normalized_data.ndim == 3
    assert not np.any(np.isnan(normalized_data))

    return normalized_data


def prepare_inputs_targets(len_time, input_gap, input_length, output_length, pred_shift, samples_gap):
    """
    Prepare sliding-window indices for sequence-to-sequence prediction.
    """
    assert output_length >= pred_shift

    input_span = input_gap * (input_length - 1) + 1
    pred_gap = output_length // pred_shift

    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, output_length, pred_gap) + input_span + pred_gap - 1

    indices = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_shift)

    max_n_sample = len_time - (input_span + output_length - 1)
    indices = indices + np.arange(max_n_sample)[:, np.newaxis] @ np.ones(
        (1, input_length + pred_shift), dtype=int
    )

    if samples_gap > 1:
        indices = indices[::samples_gap]

    return indices


class tp_dataset(Dataset):
    """
    Dataset for precipitation forecasting with auxiliary meteorological variables.

    Input tensor shape:
        [time, channels, lat, lon]

    Returned sample shape:
        input_mm  : [Tin,  C, lon, lat]
        target_mm : [Tout, C, lon, lat]
    """

    def __init__(self, tp, samples_gap=1):
        super().__init__()

        if isinstance(tp, np.ndarray):
            tp = torch.from_numpy(tp).float()
        elif torch.is_tensor(tp):
            tp = tp.float()
        else:
            raise TypeError("tp must be a numpy array or a torch tensor.")

        assert tp.ndim == 4, "Input data must have shape [time, channels, lat, lon]."

        self.tp = tp
        self.samples_gap = samples_gap
        self.indices = prepare_inputs_targets(
            len_time=self.tp.shape[0],
            input_gap=config.input_gap,
            input_length=config.input_length,
            output_length=config.output_length,
            pred_shift=config.pred_shift,
            samples_gap=self.samples_gap,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx_combined = self.indices[idx]

        input_indices = idx_combined[:config.input_length]
        target_indices = idx_combined[config.input_length:]

        input_data = self.tp[input_indices]
        target_data = self.tp[target_indices]

        # [T, C, lat, lon] -> [T, C, lon, lat]
        input_mm = input_data.permute(0, 1, 3, 2).contiguous()
        target_mm = target_data.permute(0, 1, 3, 2).contiguous()

        return input_mm, target_mm