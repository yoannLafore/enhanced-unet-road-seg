import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
    ):
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
    """Perform /2 downsampling using strided convolutions"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # TODO : do we relu here ?

    def forward(self, x):
        return self.down(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.down = DownBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x_down, x


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
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
        super().__init__()

        self.up = UpBlock(in_channels, out_channels_up)
        self.conv = DoubleConv(out_channels_up + in_channels_skip, out_channels_conv)

    def forward(self, x, skip):
        x = self.up(x)

        # Interpolate in case of different sizes due
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.concat([skip, x], dim=1)

        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.5):
        super().__init__()

        self.enc1 = Encoder(in_channels, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        self.bottom_conv = DoubleConv(512, 1024)
        self.bottom_conv2 = DoubleConv(1024, 1024)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.dec1 = Decoder(1024, 512, 512, 512)
        self.dec2 = Decoder(512, 256, 256, 256)
        self.dec3 = Decoder(256, 128, 128, 128)
        self.dec4 = Decoder(128, 64, 64, 64)

        self.conv_map = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, get_features=False):
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


class ResNet50Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Encoder
        self.enc0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        self.maxpool = resnet.maxpool

        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Decoder
        self.dec4 = Decoder(2048, 1024, 1024, 1024)
        self.dec3 = Decoder(1024, 512, 512, 512)
        self.dec2 = Decoder(512, 256, 256, 256)
        self.dec1 = Decoder(256, 64, 64, 64)

        self.conv_map = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        x_in = x

        skip0 = self.enc0(x)
        x = self.maxpool(skip0)

        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        x = self.enc4(skip3)

        x = self.dec4(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec2(x, skip1)
        x = self.dec1(x, skip0)

        if x.shape[2:] != x_in.shape[2:]:
            x = F.interpolate(
                x, size=x_in.shape[2:], mode="bilinear", align_corners=False
            )

        logits = self.conv_map(x)
        preds = torch.sigmoid(logits)

        return logits, preds
