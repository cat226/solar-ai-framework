"""Shared class-order remapping for MobileNet training/evaluation datasets.

Used by both train_mobilenet.py and evaluate_mobilenet.py so the fix for
ImageFolder.__getitem__ reading from self.samples (not self.targets) lives in
exactly one place, not two copies that could drift apart again.
"""
from __future__ import annotations

from torch.utils.data import Dataset
from torchvision import datasets


class RemappedImageFolder(Dataset):
    """Wraps an ImageFolder, remapping its alphabetical class indices to ``classes`` order.

    ``ImageFolder.__getitem__`` reads labels from ``self.samples``, not
    ``self.targets`` - reassigning ``.targets`` alone has no effect on what a
    DataLoader actually yields. This wrapper remaps the target at
    ``__getitem__`` time instead, so training/evaluation genuinely see
    production-order labels.
    """

    def __init__(self, wrapped: datasets.ImageFolder, classes: list[str]):
        self._wrapped = wrapped
        self.classes = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self._mapping = {wrapped.class_to_idx[name]: classes.index(name) for name in classes}
        self.targets = [self._mapping[t] for t in wrapped.targets]

    def __len__(self) -> int:
        return len(self._wrapped)

    def __getitem__(self, idx: int):
        sample, original_target = self._wrapped[idx]
        return sample, self._mapping[original_target]


def remap_to_class_order(ds: datasets.ImageFolder, classes: list[str]) -> RemappedImageFolder:
    """Validate ``ds`` covers exactly ``classes`` and wrap it in production order.

    Raises:
        RuntimeError: If ``ds``'s classes (any order) don't exactly match ``classes``.
    """
    if sorted(ds.classes) != sorted(classes):
        raise RuntimeError(f"dataset classes must exactly equal {classes}; got {ds.classes}")
    return RemappedImageFolder(ds, classes)
