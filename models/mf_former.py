import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import mit_b2
from .MS_UNet import DoubleDSC


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ConvCompress(nn.Module):
    """
    Compress auxiliary meteorological variables with 3D convolution.

    Input:
        [B, T, C, H, W]
    Output:
        [B, T, H, W]
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        normalization=True,
        activation=True,
    ):
        super().__init__()

        self.layer1 = self._make_layer(
            in_channels, mid_channels, kernel_size, stride, padding, normalization, activation
        )
        self.layer2 = self._make_layer(
            mid_channels, out_channels, kernel_size, stride, padding, normalization, activation
        )

    @staticmethod
    def _make_layer(in_channels, out_channels, kernel_size, stride, padding, normalization, activation):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]

        if normalization:
            layers.append(nn.BatchNorm3d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.layer1(x)
        x = self.layer2(x)

        # [B, 1, T, H, W] -> [B, T, 1, H, W] -> [B, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).squeeze(2)
        return x


class InverseOverlapPatchEmbed(nn.Module):
    """Inverse patch embedding used in the decoder."""

    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        patch_size = (patch_size, patch_size)

        self.proj = nn.ConvTranspose2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(1, 1),
            output_padding=1,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ConvModule(nn.Module):
    """Simple conv-bn-relu block."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MixedFusionBlock(nn.Module):
    """
    Multi-modal fusion block with three element-wise operators:
    sum, product, and max.
    """

    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            act_fn,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

    def forward(self, x_tp, x_mm):
        residual = x_tp

        fusion_sum = x_tp + x_mm
        fusion_mul = x_tp * x_mm

        modal_in1 = x_tp.unsqueeze(1)
        modal_in2 = x_mm.unsqueeze(1)
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        fused = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(fused)
        out2 = self.layer2(torch.cat((out1, residual), dim=1))
        return out2


class SegFormerDecoder(nn.Module):
    """U-shaped decoder for multi-scale feature aggregation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dims=[64, 128, 320],
        dropout_ratio=0.1,
    ):
        super().__init__()

        self.inverse_patch_embed1 = InverseOverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_chans=in_channels[3],
            embed_dim=out_channels[0],
        )
        self.fuse1 = ConvModule(c1=embed_dims[2] * 2, c2=embed_dims[2])

        self.inverse_patch_embed2 = InverseOverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_chans=in_channels[2],
            embed_dim=out_channels[1],
        )
        self.fuse2 = ConvModule(c1=embed_dims[1] * 2, c2=embed_dims[1])

        self.inverse_patch_embed3 = InverseOverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_chans=in_channels[1],
            embed_dim=out_channels[2],
        )
        self.fuse3 = ConvModule(c1=embed_dims[0] * 2, c2=embed_dims[0])

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dims[0], 6, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.inverse_patch_embed1(c4)
        _c4 = _c4.reshape(n, h * 2, w * 2, -1).permute(0, 3, 1, 2).contiguous()
        _c4 = self.fuse1(torch.cat((_c4, c3), dim=1))

        _c3 = self.inverse_patch_embed2(_c4)
        _c3 = _c3.reshape(n, h * 4, w * 4, -1).permute(0, 3, 1, 2).contiguous()
        _c3 = self.fuse2(torch.cat((_c3, c2), dim=1))

        _c2 = self.inverse_patch_embed3(_c3)
        _c2 = _c2.reshape(n, h * 8, w * 8, -1).permute(0, 3, 1, 2).contiguous()
        _c2 = self.fuse3(torch.cat((_c2, c1), dim=1))

        x = self.dropout(_c2)
        x = self.linear_pred(x)
        return x


class SegFormer(nn.Module):
    """
    MF-Former main model.

    Inputs:
        [B, T, 5, H, W]
    where channel 0 is precipitation and channels 1:5 are auxiliary variables.
    """

    def __init__(self, num_classes=6, phi="b2", pretrained=False):
        super().__init__()

        self.in_channels = {"b2": [64, 128, 320, 512]}[phi]
        self.out_channels = {"b2": [320, 128, 64]}[phi]

        self.backbone = {"b2": mit_b2}[phi](pretrained)
        self.backbone_mm = {"b2": mit_b2}[phi](pretrained)

        self.decoder = SegFormerDecoder(self.in_channels, self.out_channels)

        self.conv_3d = ConvCompress(
            in_channels=4,
            mid_channels=1,
            out_channels=1,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )

        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.conv_1 = DoubleDSC(64, 64)
        self.conv_2 = DoubleDSC(128, 128)
        self.conv_3 = DoubleDSC(320, 320)
        self.conv_4 = DoubleDSC(512, 512)

        self.fu_1 = MixedFusionBlock(64, 64, act_fn)
        self.fu_2 = MixedFusionBlock(128, 128, act_fn)
        self.fu_3 = MixedFusionBlock(320, 320, act_fn)
        self.fu_4 = MixedFusionBlock(512, 512, act_fn)

    def forward(self, inputs):
        # precipitation branch
        x_tp = inputs[:, :, 0, :, :]

        # auxiliary meteorological branch
        x_mm = self.conv_3d(inputs[:, :, 1:5, :, :])

        h, w = x_tp.size(2), x_tp.size(3)

        x_tp1, x_tp2, x_tp3, x_tp4 = self.backbone(x_tp)
        x_mm1, x_mm2, x_mm3, x_mm4 = self.backbone_mm(x_mm)

        x_s1 = self.conv_1(self.fu_1(x_tp1, x_mm1))
        x_s2 = self.conv_2(self.fu_2(x_tp2, x_mm2))
        x_s3 = self.conv_3(self.fu_3(x_tp3, x_mm3))
        x_s4 = self.conv_4(self.fu_4(x_tp4, x_mm4))

        x = self.decoder([x_s1, x_s2, x_s3, x_s4])
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        return x