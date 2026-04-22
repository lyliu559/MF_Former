from torch import nn


def ini_model_params(model, ini_params_mode):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            if ini_params_mode == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif ini_params_mode == "orthogonal":
                nn.init.orthogonal_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)