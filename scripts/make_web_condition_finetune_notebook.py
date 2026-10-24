"""Generate kaggle/web_condition_finetune.ipynb — GPU full fine-tune of
ConvNeXt-Tiny on the web-only 4-class condition pool (web_condition_v2).

Protocol (identical folds to scripts/eval_backbones_grouped_cv.py, shipped as
`folds.csv` in the dataset so scikit-learn version differences cannot change
them): 5-fold family-grouped CV -> every image predicted once out-of-fold with a
model that never saw its family; then ONE final model on all images. The epoch
count is fixed up-front and the headline is the LAST epoch (no selection on the
held-out fold). Reference numbers for the frozen-embedding logistic regression
are printed next to the fine-tuned ones.

No first-party / rooftop image is part of this dataset.

Regenerable: `python scripts/make_web_condition_finetune_notebook.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle" / "web_condition_finetune.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").splitlines()]}


CELLS = [
    md("# Solar AI - web-only condition classifier: ConvNeXt-Tiny fine-tune (v2)",
       "",
       "Research only. Attach the private dataset `edithstark/solar-ai-web-condition-v2` "
       "(2,618 deduplicated web photos, 4 classes, family-grouped `folds.csv`). Accelerator = GPU, Internet = on "
       "(pretrained ConvNeXt weights). Run All.",
       "",
       "Web photos only; licences of all three sources are unresolved. The result is a web-distribution estimate, "
       "NOT a rooftop-camera accuracy."),
    md("## 1. Environment"),
    code(r'''
import sys, platform, subprocess, json, os, time, hashlib, math, random
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.7.1', 'torchvision==0.22.1'], check=True)
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
assert torch.cuda.is_available(), 'CUDA unavailable - refusing to run the CPU fallback.'
ENV = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
           cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
print(json.dumps(ENV, indent=2))
'''),
    md("## 2. Locate the dataset"),
    code(r'''
CLASSES = ['CLEAN', 'DUSTY', 'BIRD_DROP', 'PHYSICAL_DAMAGE']   # canonical order = class index everywhere
hits = sorted(Path('/kaggle/input').rglob('folds.csv'))
assert hits, 'dataset with folds.csv not attached'
DATA = hits[0].parent
df = pd.read_csv(DATA / 'folds.csv')
assert set(df.label) == set(CLASSES), set(df.label)
df['y'] = df.label.map({c: i for i, c in enumerate(CLASSES)})
print(DATA, len(df), df.label.value_counts().to_dict(), 'folds:', sorted(df.fold.unique()))
# a family must never sit in two folds
assert df.groupby('family').fold.nunique().max() == 1, 'family leaks across folds'
'''),
    md("## 3. Data (images cached in RAM at <=448px long side) and model"),
    code(r'''
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
SEED, SIZE, EPOCHS, BATCH, LR = 42, 288, 12, 32, 1e-4
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def load_small(p):
    with Image.open(p) as im:
        im = im.convert('RGB'); im.thumbnail((448, 448), Image.LANCZOS); return im.copy()
t0 = time.time()
IMGS = [load_small(DATA / r) for r in df.out_rel]
print(f'cached {len(IMGS)} images in {time.time() - t0:.0f}s')

tf_train = transforms.Compose([transforms.RandomResizedCrop((SIZE, SIZE), scale=(0.5, 1.0), ratio=(0.8, 1.6)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.25, 0.25, 0.2, 0.02),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tf_eval = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(self, idx, tf): self.idx, self.tf = list(idx), tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; return self.tf(IMGS[j]), int(df.y.iloc[j])

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_model(train_idx, tag):
    seed_all(SEED)
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = nn.Linear(768, len(CLASSES)); m = m.cuda()
    cnt = np.bincount(df.y.iloc[train_idx], minlength=len(CLASSES))
    w = torch.tensor((len(train_idx) / (len(CLASSES) * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32).cuda()  # sqrt-inverse freq
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    dl = DataLoader(DS(train_idx, tf_train), batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler('cuda')
    for ep in range(1, EPOCHS + 1):
        m.train(); tl = n = 0; t1 = time.time()
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = crit(m(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss) * len(y); n += len(y)
        print(f'[{tag}] epoch {ep}/{EPOCHS} loss {tl / n:.3f} ({time.time() - t1:.0f}s)', flush=True)
    return m

@torch.no_grad()
def predict(m, idx):
    m.eval(); out = []
    dl = DataLoader(DS(idx, tf_eval), batch_size=64, shuffle=False, num_workers=2)
    for x, _ in dl:
        x = x.cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            p = (torch.softmax(m(x).float(), 1) + torch.softmax(m(torch.flip(x, dims=[3])).float(), 1)) / 2   # + hflip TTA
        out.append(p.cpu().numpy())
    return np.concatenate(out)
'''),
    md("## 4. 5-fold family-grouped CV (last epoch is the headline; no selection on the held-out fold)"),
    code(r'''
oof = np.zeros((len(df), len(CLASSES)))
fold_times = {}
for f in sorted(df.fold.unique()):
    t0 = time.time()
    tr = np.where(df.fold != f)[0]; te = np.where(df.fold == f)[0]
    m = train_model(tr, f'fold{f}')
    oof[te] = predict(m, te)
    fold_times[int(f)] = time.time() - t0
    acc = float((oof[te].argmax(1) == df.y.iloc[te].values).mean())
    print(f'### fold {f}: n_test={len(te)} acc={acc:.3f}  ({fold_times[int(f)] / 60:.1f} min)', flush=True)
    del m; torch.cuda.empty_cache()
OUT = Path('/kaggle/working/output'); OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / 'oof_probs.npy', oof)
'''),
    md("## 5. Out-of-fold metrics vs the frozen-embedding reference"),
    code(r'''
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]
def summarize(y, pred, lin):
    k = len(CLASSES); cm = np.zeros((k, k), int)
    for t, p in zip(y, pred): cm[t, p] += 1
    per, f1s, recs = {}, [], []
    for i, c in enumerate(CLASSES):
        tp, nt, npred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        pr = tp / npred if npred else 0.0; rc = tp / nt if nt else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr and rc else 0.0
        per[c] = dict(n=int(nt), recall=round(rc, 3), recall_wilson95=wilson(int(tp), int(nt)), precision=round(pr, 3), f1=round(f1, 3))
        f1s.append(f1); recs.append(rc)
    by_lin = {l: dict(n=int((lin == l).sum()), accuracy=round(float((y[lin == l] == pred[lin == l]).mean()), 3)) for l in sorted(set(lin))}
    return dict(accuracy=round(float((y == pred).mean()), 3), balanced_accuracy=round(float(np.mean(recs)), 3),
                macro_f1=round(float(np.mean(f1s)), 3), per_class=per, confusion_matrix=cm.tolist(), by_lineage=by_lin)
y = df.y.values; S = summarize(y, oof.argmax(1), df.lineage.values)
# reference: frozen convnext_tiny@320 + balanced logistic regression, same folds (datasets/.../backbone_grouped_cv.json)
REF = dict(accuracy=0.830, balanced_accuracy=0.792, macro_f1=0.790,
           recall={'CLEAN': 0.876, 'DUSTY': 0.763, 'BIRD_DROP': 0.758, 'PHYSICAL_DAMAGE': 0.771})
print(f"fine-tuned  acc={S['accuracy']} balacc={S['balanced_accuracy']} macroF1={S['macro_f1']}")
print(f"frozen LR   acc={REF['accuracy']} balacc={REF['balanced_accuracy']} macroF1={REF['macro_f1']}  (reference)")
for c in CLASSES:
    m = S['per_class'][c]; print(f"  {c:16s} n={m['n']:4d} recall={m['recall']} wilson95={m['recall_wilson95']} precision={m['precision']}  | frozen recall {REF['recall'][c]}")
print('confusion (rows=true, cols=pred, order', CLASSES, ')'); [print('  ', r) for r in S['confusion_matrix']]
print('by lineage:', S['by_lineage'])
(OUT / 'cv_metrics.json').write_text(json.dumps(dict(finetuned=S, frozen_lr_reference=REF, env=ENV, epochs=EPOCHS, size=SIZE, lr=LR,
                                                    batch=BATCH, seed=SEED, fold_minutes={k: round(v / 60, 2) for k, v in fold_times.items()}), indent=2))
'''),
    md("## 6. Final model on ALL images (same recipe, fixed epochs) + manifest"),
    code(r'''
t0 = time.time()
final = train_model(np.arange(len(df)), 'final')
torch.save(final.state_dict(), OUT / 'convnext_tiny_condition_web_v2.pth')
sha = hashlib.sha256((OUT / 'convnext_tiny_condition_web_v2.pth').read_bytes()).hexdigest()
(OUT / 'training_manifest.json').write_text(json.dumps(dict(
    classes_index_order=CLASSES, architecture='torchvision convnext_tiny, classifier[2]=Linear(768,4), ImageNet-1k pretrained',
    preprocessing=f'Resize(({SIZE},{SIZE})) whole frame, ImageNet mean/std; eval = mean of image and horizontal flip',
    epochs=EPOCHS, batch=BATCH, lr=LR, optimizer='AdamW wd=0.05 + OneCycle', loss='CE, sqrt-inverse-frequency weights, label_smoothing=0.1',
    amp='fp16 autocast', seed=SEED, n_train=len(df), env=ENV, final_train_minutes=round((time.time() - t0) / 60, 2),
    checkpoint_sha256=sha, status='RESEARCH ONLY - web photos, never validated on the target rooftop camera'), indent=2))
print('checkpoint sha256', sha)
print(sorted(p.name for p in OUT.iterdir()))
'''),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB)
