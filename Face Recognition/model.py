"""
Face recognition models.

Two model paths are supported:

1. `iresnet{18,34,50,100}` — improved-ResNet backbone (InsightFace style),
   trainable from scratch with the ArcFace head defined here. This is what
   `train.py` uses by default. Output: 512-dim embedding.

2. `facenet_vggface2` — pretrained InceptionResnetV1 (facenet-pytorch),
   trained on VGGFace2 by Tim Esler. Used by `inference.py` when no
   trained checkpoint is available, so the system works out-of-the-box
   with high accuracy on real faces. Output: 512-dim embedding.

The iResNet block follows Deng et al., "ArcFace: Additive Angular Margin
Loss for Deep Face Recognition" (CVPR 2019), using the BN-Conv-BN-PReLU-
Conv-BN ordering (a.k.a. IR / improved-residual block).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# iResNet (Improved ResNet) backbone
# ---------------------------------------------------------------------------

class IBasicBlock(nn.Module):
    """Improved residual block: BN -> Conv -> BN -> PReLU -> Conv -> BN."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-5)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out + identity


class IResNet(nn.Module):
    """iResNet for face embedding. Input: (B, 3, 112, 112). Output: (B, embedding_dim)."""

    def __init__(
        self,
        block_counts: Sequence[int],
        embedding_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.in_planes = 64
        # Stem: 3x3 stride 1 (preserves 112x112 spatial size for first stage)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        # Channel progression: 64 -> 128 -> 256 -> 512
        # Spatial progression: 112 -> 56 -> 28 -> 14 -> 7
        self.layer1 = self._make_layer(64, block_counts[0], stride=2)
        self.layer2 = self._make_layer(128, block_counts[1], stride=2)
        self.layer3 = self._make_layer(256, block_counts[2], stride=2)
        self.layer4 = self._make_layer(512, block_counts[3], stride=2)

        # Embedding head: BN -> Dropout -> Flatten -> Linear -> BN
        # Spatial size after layer4 is 7x7 for a 112x112 input.
        self.bn_final = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=1e-5)
        # Common ArcFace trick: freeze the affine on the final BN so the
        # embedding norm stays well-behaved.
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IBasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None and m.weight.requires_grad:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn_final(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


_IRESNET_VARIANTS = {
    "iresnet18":  [2, 2, 2, 2],
    "iresnet34":  [3, 4, 6, 3],
    "iresnet50":  [3, 4, 14, 3],
    "iresnet100": [3, 13, 30, 3],
}


def build_iresnet(name: str = "iresnet50", embedding_dim: int = 512, dropout: float = 0.4) -> IResNet:
    if name not in _IRESNET_VARIANTS:
        raise ValueError(f"Unknown iResNet variant: {name}. Choose from {list(_IRESNET_VARIANTS)}")
    return IResNet(_IRESNET_VARIANTS[name], embedding_dim=embedding_dim, dropout=dropout)


# ---------------------------------------------------------------------------
# CrossEntropy classification head (Stage-1 warmup)
# ---------------------------------------------------------------------------

class CosineHead(nn.Module):
    """Normalized cosine classifier (a.k.a. NormFace / scaled cosine softmax).

    logits = scale * (normalize(emb) @ normalize(W).T)

    Used as the Stage-1 warmup head. Two crucial properties for a clean
    handoff to ArcFace in Stage 2:
      1. Same geometry: features and class weights live on the unit
         hypersphere, so the Stage-1 weights are the right initialization
         for Stage-2 ArcFace.
      2. No additive margin yet: training is a plain CE on cos(theta) * scale,
         so the model can actually decrease loss from random init (unlike
         ArcFace, where cos(theta + 0.5) for the target is -0.48 at
         orthogonal init -> the classifier prefers wrong classes).

    The scale (a.k.a. temperature) is the same s used in ArcFace (default 64).
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 64.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb_n = F.normalize(embeddings, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        return F.linear(emb_n, w_n) * self.scale


# ---------------------------------------------------------------------------
# ArcFace classification head (Stage-2 metric learning)
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """Additive Angular Margin Loss head.

    Reference: ArcFace, Deng et al. CVPR 2019.
        L = -log( exp(s * cos(theta_y + m)) /
                 (exp(s * cos(theta_y + m)) + sum_{j != y} exp(s * cos(theta_j))) )

    Notes on numerical stability:
    - cos(theta+m) is computed via the angle-addition identity to avoid an
      explicit acos that can NaN at boundaries.
    - For a small set of "easy" cases where theta+m crosses pi (cos(theta) <
      cos(pi-m)), we fall back to cos(theta) - m * sin(m), which preserves
      monotonicity and matches the official InsightFace implementation.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Margin can be updated mid-training via `set_margin` to support
        # gradual margin warmup (margin = 0 -> target over the first few
        # epochs of Stage 2). This avoids the "loss explodes when ArcFace
        # turns on" failure mode where pre-margin cos(theta_y) is too low
        # for the model to absorb the full angular penalty in one step.
        self.set_margin(margin)

    def set_margin(self, margin: float) -> None:
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize features and class weights, then dot-product = cos(theta).
        emb_n = F.normalize(embeddings, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(emb_n, w_n).clamp(-1 + 1e-7, 1 - 1e-7)

        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits = torch.where(one_hot.bool(), cos_theta_m, cos_theta) * self.scale
        return logits


# ---------------------------------------------------------------------------
# Pretrained-weights wrapper (FaceNet / VGGFace2)
# ---------------------------------------------------------------------------

class FaceNetVGGFace2Wrapper(nn.Module):
    """InceptionResnetV1 pretrained on VGGFace2 (via facenet-pytorch).

    Wrapped so the same `.embed(face_tensor) -> (B, 512)` interface works
    for both this and our iResNet. Used as the default inference model
    when no trained ArcFace checkpoint is on disk.
    """

    def __init__(self):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1  # type: ignore
        except ImportError as e:
            raise ImportError(
                "facenet-pytorch is required for the pretrained VGGFace2 path. "
                "Install with: pip install facenet-pytorch"
            ) from e
        self.net = InceptionResnetV1(pretrained="vggface2").eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# High-level builder used by inference / quantization scripts
# ---------------------------------------------------------------------------

def build_embedding_model(name: str, embedding_dim: int = 512) -> nn.Module:
    if name == "facenet_vggface2":
        return FaceNetVGGFace2Wrapper()
    return build_iresnet(name, embedding_dim=embedding_dim)


def load_checkpoint(model: nn.Module, path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
