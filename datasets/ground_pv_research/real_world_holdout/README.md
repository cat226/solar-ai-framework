# Real-world rooftop holdout — ground PV research detector

**Qualitative evaluation only. Not training data. No fabricated ground truth.**

These are genuine close-up rooftop photographs supplied by the project
owner as examples of the actual target domain (close-up panels, perspective
distortion, partial panels, rooftop backgrounds, visible bird-dropping
contamination, clean panels, varied orientations).

## Rules

- **Never** added to public training / validation / test data.
- Used only for qualitative inference (`scripts/infer_ground_pv.py`).
- No TP / FP / FN / precision / recall / mAP is reported for these images
  unless a human annotates verified ground truth. Predictions are classified
  qualitatively only (plausible / likely false positive / missed defect /
  localization problem / correct panel localization / uncertain) in
  `datasets/ground_pv_research/reports/real_world_holdout.md`.

## Provenance

| field | value |
|---|---|
| supplied by | project owner (Ramana Sree), in-conversation, 2026-09-07 |
| original location | `E:\Solar Real Images\` (6 × `WhatsApp Image 2026-09-07 at 15.25.xx.jpeg`) |
| count | 6 |
| capture | handheld phone photos of real rooftop PV installations |

### Status note (2026-09-07)

At the time this pipeline was built, the `E:` drive that holds
`E:\Solar Real Images\` was **not mounted** on the dev machine, so the six
files could not be copied into `images/` here. Populate this folder when
`E:` is reattached:

```bash
cp "E:/Solar Real Images/"*.jpeg datasets/ground_pv_research/real_world_holdout/images/
```

`images/` is gitignored — the photos are never committed. Once populated,
record each file's SHA-256 in `MANIFEST.json` in this folder.
