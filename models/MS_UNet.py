import torch
import torch.nn as nn

from configs import config_tp as config


class UNet_HR(nn.Module):
    """U-shaped multi-scale feature extraction module."""

    def __init__(self, kernels_per_layer=2, bilinear=True):
        super().__init__()

        self.in_seq_len = config.in_seq_len
        self.out_seq_len = config.out_seq_len
        self.bilinear = bilinear

        self.dc = DoubleDSC1(self.in_seq_len, 64, kernels_per_layer=kernels_per_layer)

        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.down2 = DownDS(128, 320, kernels_per_layer=kernels_per_layer)
        self.down3 = DownDS(320, 512, kernels_per_layer=kernels_per_layer)
        self.down4 = DownDS(512, 1024, kernels_per_layer=kernels_per_layer)

        self.tr11 = TransUp(128, 64, kernels_per_layer=kernels_per_layer)
        self.tr12 = TransDown(64, 128, kernels_per_layer=kernels_per_layer)

        self.tr21 = TransUp(320, 128, kernels_per_layer=kernels_per_layer)
        self.tr22 = TransDown(128, 320, kernels_per_layer=kernels_per_layer)

        self.tr31 = TransUp(512, 320, kernels_per_layer=kernels_per_layer)
        self.tr32 = TransDown(320, 512, kernels_per_layer=kernels_per_layer)

        self.tr41 = TransUp(1024, 512, kernels_per_layer=kernels_per_layer)
        self.tr42 = TransDown(512, 1024, kernels_per_layer=kernels_per_layer)

        self.up4 = UpDS(1024, 512, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(512, 320, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(320, 128, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up1 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.out_seq_len)

    def forward(self, x):
        # encoder
        x = self.dc(x)
        x1 = self.down1(x)
        xtr11 = self.tr11(x, x1)
        x1tr12 = self.tr12(x, x1)

        x2 = self.down2(x1tr12)
        x1tr21 = self.tr21(x1tr12, x2)
        x2tr22 = self.tr22(x1tr12, x2)

        x3 = self.down3(x2tr22)
        x2tr31 = self.tr31(x2tr22, x3)
        x3tr32 = self.tr32(x2tr22, x3)

        x4 = self.down4(x3tr32)
        x3tr41 = self.tr41(x3tr32, x4)
        x4tr42 = self.tr42(x3tr32, x4)

        # decoder
        x = self.up4(x3tr41, x4tr42)
        x = self.up3(x2tr31, x)
        x = self.up2(x1tr21, x)
        x = self.up1(xtr11, x)
        x = self.outc(x)

        return [xtr11, x1tr21, x2tr31, x3tr41]


class DoubleDSC1(nn.Module):
    """Initial downsampling + double depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=True)

        self.dsc_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                out_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.dsc_conv(x)


class DoubleDSC(nn.Module):
    """Two stacked depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()

        self.dsc_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                out_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dsc_conv(x)


class DownDS(nn.Module):
    """Downsampling block: max pooling followed by double DSC."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleDSC(in_channels, out_channels, kernels_per_layer=kernels_per_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upsampling block with feature refinement."""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                DepthwiseSeparableConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    kernels_per_layer=kernels_per_layer,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(
                out_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x2_up = self.up(x2)
        x1 = self.relu(x1 + x2_up)
        x = self.conv(x1)
        x = self.relu(x + x2_up)
        return x


class TransUp(nn.Module):
    """Cross-scale top-down transformation block."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()

        self.up = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        x = self.relu(x1 + x2)
        return x


class TransDown(nn.Module):
    """Cross-scale bottom-up transformation block."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()

        self.down = nn.Sequential(
            DepthwiseSeparableConvDown(
                in_channels,
                out_channels,
                kernel_size=2,
                kernels_per_layer=kernels_per_layer,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.down(x1)
        x = self.relu(x1 + x2)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer,
            output_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConvDown(nn.Module):
    """Stride-2 depthwise separable convolution for downsampling."""

    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer,
            output_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class OutConv(nn.Module):
    """Final 1x1 convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)