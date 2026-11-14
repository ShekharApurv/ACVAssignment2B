import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FaceSegNet(nn.Module):
    """
    Modest encoder-decoder network for 19-class facial semantic segmentation.
    - Baseline (use_skip=False): ResNet-18 encoder + plain bilinear/conv decoder.
    - Controlled change (use_skip=True): Adds 1x1-projected additive skip fusion at strides 16, 8, and 4.
    """

    def __init__(self, num_classes=19, use_skip=False):
        super().__init__()
        self.use_skip = use_skip

        enc = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)  # stride 4
        self.layer1 = enc.layer1  # stride 4,  64 ch
        self.layer2 = enc.layer2  # stride 8,  128 ch
        self.layer3 = enc.layer3  # stride 16, 256 ch
        self.layer4 = enc.layer4  # stride 32, 512 ch

        self.dec4 = ConvBlock(512, 256)
        self.dec3 = ConvBlock(256, 128)
        self.dec2 = ConvBlock(128, 64)
        self.dec1 = ConvBlock(64, 32)

        if self.use_skip:
            self.skip3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256))
            self.skip2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128))
            self.skip1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

        self.head = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        e1 = self.layer1(self.stem(x))  # 64 x 64
        e2 = self.layer2(e1)             # 32 x 32
        e3 = self.layer3(e2)             # 16 x 16
        e4 = self.layer4(e3)             # 8 x 8

        d4 = self.dec4(F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False))
        if self.use_skip:
            d4 = d4 + self.skip3(e3)

        d3 = self.dec3(F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False))
        if self.use_skip:
            d3 = d3 + self.skip2(e2)

        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False))
        if self.use_skip:
            d2 = d2 + self.skip1(e1)

        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False))
        out = self.head(F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False))
        return out


def build_model(arch="baseline", num_classes=19):
    if arch == "baseline":
        return FaceSegNet(num_classes=num_classes, use_skip=False)
    elif arch == "skip_fusion":
        return FaceSegNet(num_classes=num_classes, use_skip=True)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
