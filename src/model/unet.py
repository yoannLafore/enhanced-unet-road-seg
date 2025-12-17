import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
    ):
        """Initialize DoubleConv block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
        """

        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Perform /2 downsampling using strided convolutions or maxpooling."""

    def __init__(self, in_channels, out_channels, use_maxpool=False):
        """Initialize DownBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_maxpool (bool): Whether to use max pooling instead of strided convolutions.
        """
        super().__init__()

        self.down = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            if not use_maxpool
            else nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.down(x)


class Encoder(nn.Module):
    """Encoder block for the U-Net."""

    def __init__(self, in_channels, out_channels, use_maxpool=False):
        """Initialize Encoder block for the U-Net.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_maxpool (bool): Whether to use max pooling instead of strided convolutions.
        """
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.down = DownBlock(out_channels, out_channels, use_maxpool=use_maxpool)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x_down, x


class UpBlock(nn.Module):
    """UpBlock for the U-Net."""

    def __init__(self, in_channels, out_channels):
        """Initialize UpBlock for the U-Net.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Decoder(nn.Module):

    def __init__(
        self, in_channels, out_channels_up, out_channels_conv, in_channels_skip
    ):
        """Initialize Decoder block for the U-Net.

        Args:
            in_channels (int): Number of input channels.
            out_channels_up (int): Number of output channels for the upsampling block.
            out_channels_conv (int): Number of output channels for the convolutional block.
            in_channels_skip (int): Number of input channels for the skip connection.
        """
        super().__init__()

        self.up = UpBlock(in_channels, out_channels_up)
        self.conv = DoubleConv(out_channels_up + in_channels_skip, out_channels_conv)

    def forward(self, x, skip):
        """Forward pass of the Decoder block.

        Args:
            x (Tensor): Input tensor.
            skip (Tensor): Skip connection tensor.

        Returns:
            Tensor: Output tensor after upsampling and convolution.
        """
        x = self.up(x)

        # Interpolate in case of different sizes due
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.concat([skip, x], dim=1)

        return self.conv(x)


class Unet(nn.Module):
    """U-Net model."""

    def __init__(
        self, in_channels=3, out_channels=1, dropout_prob=0.5, use_maxpool=False
    ):
        """Initialize U-Net model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout_prob (float): Dropout probability at the bottom convolutional block.
            use_maxpool (bool): Whether to use max pooling instead of strided convolutions.
        """
        super().__init__()

        self.enc1 = Encoder(in_channels, 64, use_maxpool=use_maxpool)
        self.enc2 = Encoder(64, 128, use_maxpool=use_maxpool)
        self.enc3 = Encoder(128, 256, use_maxpool=use_maxpool)
        self.enc4 = Encoder(256, 512, use_maxpool=use_maxpool)

        self.bottom_conv = DoubleConv(512, 1024)
        self.bottom_conv2 = DoubleConv(1024, 1024)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.dec1 = Decoder(1024, 512, 512, 512)
        self.dec2 = Decoder(512, 256, 256, 256)
        self.dec3 = Decoder(256, 128, 128, 128)
        self.dec4 = Decoder(128, 64, 64, 64)

        self.conv_map = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, get_features=False):
        """Forward pass of the U-Net model.

        Args:
            x (Tensor): Input tensor.
            get_features (bool): Whether to return intermediate features.

        Returns:
            Tuple[Tensor, Tensor]: Logits and predictions.
            or Tuple[Tensor, Tensor, Tensor]: Logits, predictions, and features if get_features is True.
        """
        enc1_out, skip_enc1 = self.enc1(x)
        enc2_out, skip_enc2 = self.enc2(enc1_out)
        enc3_out, skip_enc3 = self.enc3(enc2_out)
        enc4_out, skip_enc4 = self.enc4(enc3_out)

        x = self.bottom_conv(enc4_out)
        x = self.bottom_conv2(x)
        x = self.dropout(x)

        x = self.dec1(x, skip_enc4)
        x = self.dec2(x, skip_enc3)
        x = self.dec3(x, skip_enc2)
        features = self.dec4(x, skip_enc1)

        logits = self.conv_map(features)

        preds = torch.sigmoid(logits)

        if get_features:
            return logits, preds, features

        return logits, preds
