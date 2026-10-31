"""Notebook-compatible channel gate and SMP decoder/encoder combinations."""
import torch
from torch import nn
import segmentation_models_pytorch as smp


class ChannelAttention(nn.Module):
    def __init__(self, channels=20):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // 4), nn.ReLU(inplace=True),
                                nn.Linear(channels // 4, channels), nn.Sigmoid())

    def forward(self, x):
        # Pool in FP32 to keep gate energies stable under CUDA FP16 autocast.
        energy = self.avg_pool(x.float().square()).flatten(1)
        weights = self.fc(energy)
        return x * weights[:, :, None, None], weights


class LandslideModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.input_attention = ChannelAttention()
        # timm's Swin defaults to strict 224px input; explicitly support 256px
        # with runtime attention masks instead of resizing scientific inputs.
        encoder_kwargs = dict(img_size=256, strict_img_size=False) if cfg["encoder"].startswith("tu-swin_") else {}
        architectures = dict(unet=smp.Unet, unetplusplus=smp.UnetPlusPlus,
                             deeplabv3plus=smp.DeepLabV3Plus, upernet=smp.UPerNet, segformer=smp.Segformer)
        self.backbone = architectures[cfg["arch"]](encoder_name=cfg["encoder"],
            encoder_weights=cfg["encoder_weights"] if pretrained else None,
            in_channels=20, classes=1, activation=None, **encoder_kwargs)

    def forward(self, x):
        x, weights = self.input_attention(x)
        return self.backbone(x), weights


class FocalTverskyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
        self.tversky = smp.losses.TverskyLoss(mode="binary", alpha=cfg["tversky_alpha"],
                                          beta=cfg["tversky_beta"], from_logits=True)
        self.wf, self.wt = cfg["focal_weight"], cfg["tversky_weight"]

    def forward(self, logits, target):
        return self.wf * self.focal(logits.float(), target) + self.wt * self.tversky(logits.float(), target)
