# BDAPPV Provenance & License Verification (IGN subset)

Generated: 2026-09-03
Repository: https://github.com/cat226/solar-ai-framework
Branch: feat/yolo-detection-pipeline

This document exists specifically to satisfy this project's rule that a mirror is never
treated as independent provenance — every claim below is cross-checked against the
authoritative upstream record, not the mirror alone.

## Upstream (authoritative) record

- **Publication**: Kasmi, G., Saint-Drenan, Y-M., Trebosc, D., Jolivet, R., Leloux, J., Sarr, B.,
  Dubus, L. "A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays
  and installation metadata." *Scientific Data* 10, 59 (2023).
- **Preprint**: arXiv:2209.03726
- **Data record (DOI)**: 10.5281/zenodo.7358126 — https://zenodo.org/record/7358126
- **Zenodo license field**: CC BY 4.0 for the dataset; base imagery carries its own separate
  terms (Google's own imagery terms for the Google subset; IGN's Open License 2.0 for the IGN
  subset) — verified live via `https://zenodo.org/api/records/7358126` on 2026-09-03.
- **Authors' institution**: Mines Paris (per the Hugging Face profile below; not independently
  cross-checked against the paper's own affiliation line, given no full-text access at review
  time — treat this specific detail as corroborating, not sole, evidence).

## Mirror used for actual download: Hugging Face

- **URL**: https://huggingface.co/datasets/gabrielkasmi/bdappv
- **Why a mirror was used at all**: Zenodo's direct `bdappv.zip` download sustained only
  ~36-40KB/s (a ~63 hour ETA for 8.16GB), unaffected by reconnecting — an operational
  bottleneck, not a provenance concern. The HF mirror sustained ~230-600KB/s via direct
  `curl` to its `resolve/main/...` URLs.

### Evidence this mirror corresponds to the upstream record, not an independent/unverified re-upload

1. **Same maintainer account, not a third party.** The HF dataset is hosted under
   `gabrielkasmi`, the same username as the paper's first author (Gabriel Kasmi), consistent
   across the HF dataset repo, a companion `gabrielkasmi/bdappv-models` model repo, and a
   `gabrielkasmi/openpvmapper` project — all in the same solar-panel-detection research area.
2. **Explicit citation match.** The HF dataset card cites `arxiv: 2209.03726` — the exact
   preprint ID for the BDAPPV paper.
3. **Exact statistic match against the independently-verified Zenodo record**, checked
   2026-09-03:

   | Metric | Zenodo (upstream, live API) | Hugging Face (mirror) |
   |---|---|---|
   | IGN image count | 17,325 | 17,325 (via `gabrielkasmi/bdappv` README) |
   | Google image count | 28,807 | 28,408\* |
   | License | CC BY 4.0 | CC BY 4.0 (dataset card `license: cc-by-4.0`) |

   \* Minor discrepancy (28,807 vs 28,408) between the Zenodo record's raw crowdsourcing
   count and the HF ML-ready parquet row count for the Google config — plausibly explained by
   the HF version excluding a small number of images dropped during ML-dataset curation
   (e.g. corrupted files, near-duplicates). Not investigated further since **the Google
   subset is not being used for training** (see next section) — flagged here for completeness,
   not treated as a provenance red flag for the IGN subset actually in use, where the count
   matches exactly.
4. **Independent confirmation of the geographic split methodology** stated in the HF README
   ("spatial holdout by French department... do not re-split to ensure comparability with
   published results") matches the train/val/test counts (20,707 / 3,817 / 3,884) reported in
   both places.

**Conclusion**: the Hugging Face mirror is treated as the author's own redistribution of the
same Zenodo-recorded dataset, not as independent provenance — every license and content claim
made about it is anchored back to the Zenodo record above.

## Why the IGN subset (not Google) was selected for production training

Three independent reasons converge on the same conclusion:

1. **License cleanliness.** Per the Zenodo record's own metadata: Google base imagery is
   "subject to Google's terms" (separate from CC BY 4.0, and in practice this shows up as the
   Google-trained companion model checkpoints being licensed CC-BY-**NC** 4.0 in
   `gabrielkasmi/bdappv-models`), while IGN base imagery is under **Open License 2.0**
   (Etalab, a permissive French government open-data license) and the IGN-trained checkpoints
   are licensed CC BY 4.0 with no NonCommercial restriction. Using IGN avoids reopening the
   same NonCommercial-license question already worked through for Bird-Drop
   (`training/classification/LICENSE_COMPATIBILITY.md`).
2. **The dataset authors' own guidance.** The HF README explicitly states: "pooling both
   providers for training is discouraged, as domain augmentation rather than independent data
   results, conflating the distribution shift signal." The two providers are intended for
   cross-domain evaluation, not naive pooling.
3. **Provenance clarity.** IGN's row count matches the upstream Zenodo record exactly (see
   table above); the Google config's small discrepancy is an unresolved detail avoided
   entirely by not depending on it.

**Trade-off accepted**: IGN's native ground sampling distance is 20cm/pixel vs. Google's
10cm/pixel — potentially lower detail per panel. This can be revisited later if IGN alone
proves insufficient for detection quality; it will not be papered over by silently mixing in
Google imagery.

## Status

Provenance and license verification: **COMPLETE** for the IGN subset. Proceeding to schema
inspection and dataset audit (see `DATASET_SOURCES.md` and, once complete, the manifest under
`training/detection/`).
