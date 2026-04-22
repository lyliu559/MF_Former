import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import config_tp as config
from datasets.weatherbench import normalize_data, tp_dataset
from models.mf_former import SegFormer
from utils.initial import ini_model_params
from utils import evaluation as eval_utils


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @staticmethod
    def loss_tp(y_pred, y_true):
        """Training loss: RMSE + MAE."""
        mse = F.mse_loss(y_pred, y_true, reduction="mean")
        rmse = torch.sqrt(mse)
        mae = F.l1_loss(y_pred, y_true, reduction="mean")
        return rmse + mae

    def build_dataloader(self, data_array, batch_size, shuffle):
        dataset = tp_dataset(torch.from_numpy(data_array).float(), samples_gap=1)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        return dataset, dataloader

    def train(self, train_multi_data, val_multi_data):
        model = SegFormer(num_classes=6, phi="b2").to(self.device)
        ini_model_params(model, self.cfg.ini_params_mode)

        print(
            f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}  "
            f"Initialize model parameters with {self.cfg.ini_params_mode}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            betas=self.cfg.optim_betas,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.9,
            patience=3,
            min_lr=1e-6,
            eps=1e-6,
            verbose=True,
        )

        print("Loading training dataloader...")
        dataset_train, dataloader_train = self.build_dataloader(
            train_multi_data,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
        )
        print(f"Number of training samples: {len(dataset_train)}")
        del train_multi_data

        print("Loading validation dataloader...")
        dataset_val, dataloader_val = self.build_dataloader(
            val_multi_data,
            batch_size=self.cfg.valid_batch_size,
            shuffle=False,
        )
        print(f"Number of validation samples: {len(dataset_val)}")
        del val_multi_data

        os.makedirs(self.cfg.save_path, exist_ok=True)

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(self.cfg.train_max_epochs):
            model.train()
            epoch_loss = 0.0

            print(f"\nEpoch: {epoch + 1}/{self.cfg.train_max_epochs}")

            for step, (input_mm, target_mm) in enumerate(dataloader_train, start=1):
                input_mm = input_mm.to(self.device, non_blocking=True)
                target_mm = target_mm.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                output = model(input_mm)
                target_tp = target_mm[:, :, 0, :, :]
                loss = self.loss_tp(output, target_tp)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if step % self.cfg.loss_log_iters == 0:
                    print(
                        f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}  "
                        f"Train Epoch: {epoch + 1} "
                        f"[{step * self.cfg.train_batch_size}/{len(dataset_train)} "
                        f"({100.0 * step / len(dataloader_train):.0f}%)]\t"
                        f"Loss: {loss.item():.6f}"
                    )

            avg_train_loss = epoch_loss / len(dataloader_train)

            rmse, mae = eval_utils.valid(model, dataloader=dataloader_val)
            val_loss = rmse
            scheduler.step(val_loss)

            print(
                f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}  "
                f"Epoch: {epoch + 1}\t"
                f"Train Loss: {avg_train_loss:.6f}\t"
                f"Val RMSE: {rmse:.4f}\t"
                f"Val MAE: {mae:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                best_model_path = os.path.join(
                    self.cfg.save_path,
                    f"{self.cfg.model_name}.pth",
                )
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with validation RMSE: {val_loss:.4f}")

            print(f"Best Epoch: {best_epoch}\tBest RMSE: {best_val_loss:.4f}")

            if (epoch + 1) % self.cfg.model_save_fre == 0:
                epoch_model_path = os.path.join(
                    self.cfg.save_path,
                    f"{self.cfg.model_name}_epoch_{epoch + 1}.pth",
                )
                torch.save(model.state_dict(), epoch_model_path)


def load_multimodal_data(data_root, year_tag):
    """
    Load WeatherBench precipitation and auxiliary meteorological variables.

    Parameters
    ----------
    data_root : str
        Root directory of the processed dataset.
    year_tag : str
        For example: '2014_2016', '2017', '2018'.
    """
    tp_name = f"tp/total_precipitation_{year_tag}_1.40625deg.nc"
    rh_name = f"rh/relative_humidity_850_{year_tag}_1.40625deg.nc"
    t_name = f"t/temperature_850_{year_tag}_1.40625deg.nc"
    u_name = f"u/u_850_component_of_wind_{year_tag}_1.40625deg.nc"
    v_name = f"v/v_850_component_of_wind_{year_tag}_1.40625deg.nc"

    print("----- load tp data -----")
    tp = normalize_data(os.path.join(data_root, tp_name), "tp")
    print("----- load rh data -----")
    rh = normalize_data(os.path.join(data_root, rh_name), "r")
    print("----- load t data -----")
    t = normalize_data(os.path.join(data_root, t_name), "t")
    print("----- load u data -----")
    u = normalize_data(os.path.join(data_root, u_name), "u")
    print("----- load v data -----")
    v = normalize_data(os.path.join(data_root, v_name), "v")

    multi_data = np.stack((tp, rh, t, u, v), axis=1).astype(np.float32)
    del tp, rh, t, u, v

    return multi_data


if __name__ == "__main__":
    print("\nReading data...")

    data_root = config.dataset_root

    # training set: 2014-2016
    train_multi_data = load_multimodal_data(data_root, "2014_2016")

    # validation set: 2017
    val_multi_data = load_multimodal_data(data_root, "2017")

    trainer = Trainer(config)
    trainer.train(train_multi_data, val_multi_data)