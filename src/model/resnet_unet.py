from src.model.unet import *


class ResNetUnet(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        from_existing_resnet=None,
        weights=None,
        *args,
        **kwargs,
    ):
        """
        ResNet-Unet model.
        Args:
            backbone (str): ResNet backbone type. One of 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
            from_existing_resnet (nn.Module, optional): Existing ResNet model to use for encoder weights.
            weights (torchvision.models.ResNetWeights, optional): Pretrained weights for the ResNet backbone.
        """

        super().__init__()

        # Load ResNet backbone
        if backbone == "resnet18":
            resnet = models.resnet18(weights=weights)
            enc_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=weights)
            enc_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=weights)
            enc_channels = [64, 256, 512, 1024, 2048]
        elif backbone == "resnet101":
            resnet = models.resnet101(weights=weights)
            enc_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        if from_existing_resnet is not None:
            print("Using existing ResNet model for encoder weights")
            resnet = from_existing_resnet

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
        self.dec4 = Decoder(
            enc_channels[4], enc_channels[3], enc_channels[3], enc_channels[3]
        )
        self.dec3 = Decoder(
            enc_channels[3], enc_channels[2], enc_channels[2], enc_channels[2]
        )
        self.dec2 = Decoder(
            enc_channels[2], enc_channels[1], enc_channels[1], enc_channels[1]
        )
        self.dec1 = Decoder(
            enc_channels[1], enc_channels[0], enc_channels[0], enc_channels[0]
        )

        self.conv_map = nn.Conv2d(enc_channels[0], 1, 1)

    def forward(self, x):
        """Forward pass of the ResNet-Unet model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            logits (torch.Tensor): Output logits of shape (B, 1, H, W).
            preds (torch.Tensor): Output predictions after sigmoid of shape (B, 1, H, W).
        """

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

        # Upsample to match input size if necessary
        if x.shape[2:] != x_in.shape[2:]:
            x = F.interpolate(
                x, size=x_in.shape[2:], mode="bilinear", align_corners=False
            )

        logits = self.conv_map(x)
        preds = torch.sigmoid(logits)

        return logits, preds


# Specific ResNet-Unet variants
class ResNet18Unet(nn.Module):
    def __init__(self, pretrained=False, from_existing_resnet=None, *args, **kwargs):
        """
        Initialize the ResNet18-Unet model.

        Args:
            pretrained (bool): If True, use ImageNet pretrained weights for the ResNet18 backbone.
            from_existing_resnet (nn.Module, optional): Existing ResNet model to use for encoder weights.
        """
        super().__init__()
        self.unet = ResNetUnet(
            backbone="resnet18",
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            from_existing_resnet=from_existing_resnet,
            *args,
            **kwargs,
        )

    def forward(self, x):
        return self.unet(x)


class ResNet34Unet(nn.Module):
    def __init__(self, pretrained=False, from_existing_resnet=None, *args, **kwargs):
        """
        Initialize the ResNet34-Unet model.

        Args:
            pretrained (bool): If True, use ImageNet pretrained weights for the ResNet34 backbone.
            from_existing_resnet (nn.Module, optional): Existing ResNet model to use for encoder weights.
        """
        super().__init__()
        self.unet = ResNetUnet(
            backbone="resnet34",
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            from_existing_resnet=from_existing_resnet,
            *args,
            **kwargs,
        )

    def forward(self, x):
        return self.unet(x)


class ResNet50Unet(nn.Module):
    def __init__(self, pretrained=False, from_existing_resnet=None, *args, **kwargs):
        """
        Initialize the ResNet50-Unet model.

        Args:
            pretrained (bool): If True, use ImageNet pretrained weights for the ResNet50 backbone.
            from_existing_resnet (nn.Module, optional): Existing ResNet model to use for encoder weights.
        """
        super().__init__()
        self.unet = ResNetUnet(
            backbone="resnet50",
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            from_existing_resnet=from_existing_resnet,
            *args,
            **kwargs,
        )

    def forward(self, x):
        return self.unet(x)


class ResNet101Unet(nn.Module):
    def __init__(self, pretrained=False, from_existing_resnet=None, *args, **kwargs):
        """
        Initialize the ResNet101-Unet model.

        Args:
            pretrained (bool): If True, use ImageNet pretrained weights for the ResNet101 backbone.
            from_existing_resnet (nn.Module, optional): Existing ResNet model to use for encoder weights.
        """
        super().__init__()
        self.unet = ResNetUnet(
            backbone="resnet101",
            weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
            from_existing_resnet=from_existing_resnet,
            *args,
            **kwargs,
        )

    def forward(self, x):
        return self.unet(x)
