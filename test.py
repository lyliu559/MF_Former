import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import config_tp as config
from datasets.weatherbench import normalize_data, tp_dataset
from models.mf_former import SegFormer
from utils import evaluation as eval_utils


def load_checkpoint(model, ckpt_path, device):
    """Load model checkpoint and handle possible DataParallel prefixes."""
    state_dict = torch.load(ckpt_path, map_location=device)

    if "module." in list(state_dict.keys())[0]:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


def test(dataset_test):
    device = torch.device(
        "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    )

    model = SegFormer(num_classes=6, phi="b2").to(device)

    ckpt_path = os.path.join(config.save_path, f"{config.model_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_checkpoint(model, ckpt_path, device)

    print(f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}  Load pretrained model successfully")
    print(f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}  Start testing on the test dataset")

    print("Loading test dataloader...")
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rmse, mae, pod, far, csi = eval_utils.valid(
        model,
        dataloader=dataloader_test,
        return_event_scores=True,
    )

    print(
        f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}  "
        f"Test: RMSE: {rmse:.4f}\tMAE: {mae:.4f}\tPOD: {pod:.4f}\tFAR: {far:.4f}\tCSI: {csi:.4f}"
    )


def load_multimodal_data(data_root, year_tag):
    """Load WeatherBench precipitation and auxiliary meteorological variables."""
    tp_name = f"tp/total_precipitation_{year_tag}_1.40625deg.nc"
    rh_name = f"rh/relative_humidity_850_{year_tag}_1.40625deg.nc"
    t_name = f"t/temperature_850_{year_tag}_1.40625deg.nc"
    u_name = f"u/u_850_component_of_wind_{year_tag}_1.40625deg.nc"
    v_name = f"v/v_850_component_of_wind_{year_tag}_1.40625deg.nc"

    print("----- load test tp data -----")
    tp = normalize_data(os.path.join(data_root, tp_name), "tp")
    print("----- load test rh data -----")
    rh = normalize_data(os.path.join(data_root, rh_name), "r")
    print("----- load test t data -----")
    t = normalize_data(os.path.join(data_root, t_name), "t")
    print("----- load test u data -----")
    u = normalize_data(os.path.join(data_root, u_name), "u")
    print("----- load test v data -----")
    v = normalize_data(os.path.join(data_root, v_name), "v")

    multi_data = np.stack((tp, rh, t, u, v), axis=1).astype(np.float32)
    del tp, rh, t, u, v

    return multi_data


if __name__ == "__main__":
    print("\nReading test data...")

    data_root = config.dataset_root

    # test set: 2018
    test_multi_data = load_multimodal_data(data_root, "2018")

    print("Processing test set...")
    dataset_test = tp_dataset(torch.from_numpy(test_multi_data).float(), samples_gap=1)

    test(dataset_test)