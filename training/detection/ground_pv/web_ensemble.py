"""training/detection/ground_pv/web_ensemble.py — load and run the FINAL web-only condition ensemble
(weights/web_condition_ensemble_v1/ensemble.json): five Kaggle-fine-tuned backbones, softmax probabilities averaged,
each member averaging the image and its horizontal flip. Shared by scripts/infer_web_ensemble.py and
services/ground_pipeline.py so there is exactly one implementation.

Class order is CANONICAL (CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE) and is read from the manifest - never assumed.
RESEARCH ONLY: trained on web photos, never validated on the target rooftop camera.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch: str, n_classes: int) -> nn.Module:
    """Same head layout the Kaggle notebooks trained (weights=None: the fine-tuned state dict is loaded after)."""
    if arch.startswith("convnext"):
        ctor = {"convnext_tiny": models.convnext_tiny, "convnext_small": models.convnext_small,
                "convnext_base": models.convnext_base}[arch]
        m = ctor(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise KeyError(f"unknown architecture {arch!r}")
    return m


class WebEnsemble:
    def __init__(self, manifest: Path):
        self.dir = manifest.parent
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes_index_order"]
        self.members = []
        for mem in meta["members"]:
            m = build_model(mem["arch"], len(self.classes))
            m.load_state_dict(torch.load(self.dir / mem["file"], map_location="cpu"))
            tf = transforms.Compose([transforms.Resize((mem["size"], mem["size"])), transforms.ToTensor(),
                                     transforms.Normalize(_MEAN, _STD)])
            self.members.append((m.eval(), tf))

    @torch.no_grad()
    def predict_proba(self, image: Image.Image) -> list[float]:
        im = ImageOps.exif_transpose(image).convert("RGB")
        acc = torch.zeros(len(self.classes))
        for m, tf in self.members:
            x = tf(im).unsqueeze(0)
            acc += (torch.softmax(m(x), 1) + torch.softmax(m(torch.flip(x, dims=[3])), 1))[0] / 2
        return (acc / len(self.members)).tolist()
