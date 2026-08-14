"""Train the six-class MobileNetV2 classifier from a prepared dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

CLASSES = ["Clean", "Dusty", "Bird-Drop", "Electrical-Damage", "Physical-Damage", "Hotspot"]


def _dataset(root: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    ds = datasets.ImageFolder(root, transform=transform)
    if sorted(ds.classes) != sorted(CLASSES):
        raise RuntimeError(f"dataset classes must exactly equal {CLASSES}; got {ds.classes}")
    mapping = {ds.class_to_idx[name]: CLASSES.index(name) for name in CLASSES}
    ds.targets = [mapping[target] for target in ds.targets]
    return ds


def build_loaders(root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train = _dataset(root / "train", train_tf)
    val = _dataset(root / "val", eval_tf)
    counts = torch.bincount(torch.tensor(train.targets), minlength=len(CLASSES)).float()
    if (counts == 0).any():
        raise RuntimeError(f"every class must have training samples; counts={counts.tolist()}")
    weights = 1.0 / counts
    sampler = WeightedRandomSampler(weights[torch.tensor(train.targets)], len(train), replacement=True)
    return (
        DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=0),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("epochs, batch-size, and learning rate must be positive")
    torch.manual_seed(args.seed)
    train_loader, val_loader = build_loaders(args.data_root, args.batch_size)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.to(device))
                pred = outputs.argmax(1).cpu()
                correct += int((pred == targets).sum())
                total += len(targets)
        acc = correct / max(total, 1)
        print(f"epoch={epoch + 1} val_accuracy={acc:.6f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    if list(model.classifier[1].weight.shape) != [len(CLASSES), model.last_channel]:
        raise RuntimeError("unexpected classifier output shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(f"saved={args.output} best_val_accuracy={best_acc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
