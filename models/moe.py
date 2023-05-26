import torch
import torch.nn as nn
from .wrn import WideResNet

class Moe1(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes=10, num_flips=2, num_sc=6, num_lorot=16) -> None:
        super().__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor, drop_rate)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.backbone.nChannels, num_classes)
        self.lorot_layer = nn.Linear(self.backbone.nChannels, num_lorot)
        self.flip_layer = nn.Linear(self.backbone.nChannels, num_flips)
        self.sc_layer = nn.Linear(self.backbone.nChannels, num_sc)
        self.gating_layer = nn.Linear(self.backbone.nChannels, 3)

    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)
    
class Nomoe(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes=10, num_flips=2, num_sc=6, num_lorot=16) -> None:
        super().__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor, drop_rate)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.backbone.nChannels, num_classes)
        self.lorot_layer = nn.Linear(self.backbone.nChannels, num_lorot)
        self.flip_layer = nn.Linear(self.backbone.nChannels, num_flips)
        self.sc_layer = nn.Linear(self.backbone.nChannels, num_sc)

    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.sc_layer(out),
            )
        return self.classifier(out)
    
class Lorot(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes=10, num_lorot=16):
        super().__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor, drop_rate)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.backbone.nChannels, num_classes)
        self.lorot_layer = nn.Linear(self.backbone.nChannels, num_lorot)
    
    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out)
            )
        return self.classifier(out)

class Moe1flip(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes=10, num_flips=2, num_lorot=16) -> None:
        super().__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor, drop_rate)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.backbone.nChannels, num_classes)
        self.lorot_layer = nn.Linear(self.backbone.nChannels, num_lorot)
        self.flip_layer = nn.Linear(self.backbone.nChannels, num_flips)
        self.gating_layer = nn.Linear(self.backbone.nChannels, 2)

    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.flip_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)
    
class Moe1sc(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes=10, num_sc=6, num_lorot=16) -> None:
        super().__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor, drop_rate)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.backbone.nChannels, num_classes)
        self.lorot_layer = nn.Linear(self.backbone.nChannels, num_lorot)
        self.sc_layer = nn.Linear(self.backbone.nChannels, num_sc)
        self.gating_layer = nn.Linear(self.backbone.nChannels, 2)

    def forward(self, x, ssl_task=True):
        out = self.backbone(x)
        if self.training and ssl_task:
            return (
                self.classifier(out),
                self.lorot_layer(out),
                self.sc_layer(out),
                self.gating_layer(out)
            )
        return self.classifier(out)