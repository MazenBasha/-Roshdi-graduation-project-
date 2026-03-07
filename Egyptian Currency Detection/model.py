"""
Lightweight MobileNet-style CNN for Egyptian Currency Classification.

Architecture: Built entirely from scratch using depthwise separable convolutions.
- All weights randomly initialized (no pretrained weights).
- Designed for mobile deployment (small model, fast inference).
- Uses depthwise separable convolutions (1/8th to 1/9th the computation of standard convs).
- Batch normalization + ReLU6 for stable training and quantization-friendly activations.
- Global average pooling (no large fully-connected layers).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBnRelu(nn.Sequential):
    """Standard Conv -> BatchNorm -> ReLU6 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    Factorizes a standard convolution into:
      1. Depthwise conv (spatial filtering per channel)
      2. Pointwise conv (1x1 channel mixing)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBnRelu(
            in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels
        )
        self.pointwise = ConvBnRelu(
            in_channels, out_channels, kernel_size=1, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual block (MobileNetV2-style).
    Expand -> Depthwise -> Project with skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Expand (pointwise)
        if expand_ratio != 1:
            layers.append(ConvBnRelu(in_channels, hidden_dim, kernel_size=1))

        # Depthwise
        layers.append(ConvBnRelu(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim
        ))

        # Project (pointwise, linear - no ReLU)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round a value to the nearest divisible number."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CurrencyMobileNet(nn.Module):
    """
    Lightweight MobileNet-style architecture for currency classification.

    Architecture overview:
        Conv2d(3, 32, 3x3, stride=2)      # 224 -> 112
        InvertedResidual blocks             # progressive downsampling
        Conv2d(last, 1280, 1x1)            # feature expansion
        GlobalAvgPool                       # spatial reduction
        Dropout + Linear(1280, num_classes) # classifier

    Total params: ~2.2M (with width_mult=1.0)
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        width_mult: float = config.MODEL_WIDTH_MULT,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        # Architecture definition: (expand_ratio, out_channels, num_blocks, stride)
        inverted_residual_setting = [
            # t, c,  n, s
            [1,  16,  1, 1],   # 112 -> 112
            [6,  24,  2, 2],   # 112 -> 56
            [6,  32,  3, 2],   # 56  -> 28
            [6,  64,  4, 2],   # 28  -> 14
            [6,  96,  3, 1],   # 14  -> 14
            [6, 160,  3, 2],   # 14  -> 7
            [6, 320,  1, 1],   # 7   -> 7
        ]

        # First layer
        first_channels = _make_divisible(32 * width_mult)
        last_channels = _make_divisible(1280 * max(1.0, width_mult))

        features: List[nn.Module] = [
            ConvBnRelu(3, first_channels, kernel_size=3, stride=2)
        ]

        # Inverted residual blocks
        in_channels = first_channels
        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(in_channels, out_channels, stride=stride, expand_ratio=t)
                )
                in_channels = out_channels

        # Last conv layer
        features.append(ConvBnRelu(in_channels, last_channels, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels, num_classes),
        )

        # Initialize weights from scratch
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for all layers (from scratch)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps before pooling (for CAM visualization)."""
        return self.features(x)


def build_model(
    num_classes: int = config.NUM_CLASSES,
    width_mult: float = config.MODEL_WIDTH_MULT,
    dropout: float = config.DROPOUT_RATE,
) -> CurrencyMobileNet:
    """Factory function to create the model."""
    model = CurrencyMobileNet(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Print model summary with parameter counts."""
    total = count_parameters(model)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total:,}")
    print(f"Estimated size (FP32): {total * 4 / 1024 / 1024:.2f} MB")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Input size: {config.IMG_CHANNELS}x{config.IMG_SIZE}x{config.IMG_SIZE}")

    # Layer-by-layer summary
    print(f"\n{'Layer':<40} {'Output Shape':<20} {'Params':>10}")
    print("-" * 72)
    dummy = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    hooks = []
    layer_info = []

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                layer_info.append((name, list(output.shape), params))
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    for name, shape, params in layer_info[:30]:  # Show first 30 layers
        name_short = name[-38:] if len(name) > 38 else name
        print(f"  {name_short:<38} {str(shape):<20} {params:>10,}")
    if len(layer_info) > 30:
        print(f"  ... and {len(layer_info) - 30} more layers")
    print("-" * 72)
    print(f"  {'TOTAL':<38} {'':<20} {total:>10,}")


if __name__ == "__main__":
    model = build_model()
    model_summary(model)

    # Verify forward pass
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    out = model(x)
    print(f"\nForward pass test: input {x.shape} -> output {out.shape}")
    assert out.shape == (2, config.NUM_CLASSES), f"Unexpected output shape: {out.shape}"
    print("Model verification passed!")
