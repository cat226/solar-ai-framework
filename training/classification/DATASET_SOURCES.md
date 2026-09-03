# Dataset Provenance Report

Generated: 2026-08-14
Repository: https://github.com/cat226/solar-ai-framework
Branch: feat/mobilenet-training-pipeline

## Summary

Six production classes are required:
1. Clean
2. Dusty
3. Bird-Drop
4. Electrical-Damage
5. Physical-Damage
6. Hotspot

## PHASE 3 RE-VERIFICATION (2026-09-03, independent re-check)

Both remaining blockers were independently re-verified today, from scratch, rather than trusting
the prior write-up:

**Bird-Drop — additional candidates checked and rejected, conclusion unchanged (BLOCKED):**
- The "PV Panel Defect Dataset" Kaggle mirrors (`alicjalena/pv-panel-defect-dataset`,
  `neurobotdata/photovoltaic-panel-defect-dataset`, 792 images, six classes incl.
  `Bird-drop`/`Snow-Covered`) all trace to the same "Afroz (2023)" Kaggle source
  (`pythonafroz/solar-panel-images`) already rejected in this document for "License: Unknown" —
  independently confirmed by the SolarFCD paper's own literature review, which separately flags
  this exact source as "permission uncertain." Re-uploading under a different Kaggle username
  does not establish a valid license.
- `gitenavnath/solar-augmented-dataset` (Kaggle, ~3,000 images, includes Bird-drop): described as
  an *augmented* dataset (rotation/shift/shear/zoom applied) with no license stated for the
  underlying images — almost certainly a derivative of the same Afroz base set inflated via
  augmentation. Even setting the license question aside, pre-augmented data cannot be used as
  raw source input without violating this project's split policy (augmented copies must stay
  confined to the training split, which requires knowing which images are augmented copies of
  which originals — not knowable from a pre-augmented download).
- Multiple Roboflow "bird drop" object-detection projects show CC BY 4.0 license badges, but a
  Roboflow uploader's self-applied license badge on their platform export does not establish that
  they held the rights to relicense the underlying images — most use the same class taxonomy as
  the Afroz set (Clean/Dusty/Bird-drop/Snow-Covered), suggesting re-hosted derivatives rather than
  independent collections. Direct verification of original provenance was blocked (403) on the
  project page checked. None were accepted absent that verification.
- Mendeley "Thermal Imaging Dataset for Hotspot Detection on Solar Panels: Impact of Bird
  Droppings on Efficiency" (TRSAI, Egypt, CC BY 4.0, 850 thermal images): genuinely licensed
  and downloadable, but thermal-only (wrong modality vs. our RGB requirement) and does not
  clearly separate a discrete "bird dropping visible on panel" class from generic hotspot
  images — it studies bird droppings as a *cause* of thermal hotspots, not as a distinct
  visual class.
- "Plant_Campania" dataset (Di Tommaso et al., arXiv 2111.11709, 1500 RGB images, six defect
  classes reportedly including bird dropping): the only candidate found with a plausible,
  genuinely field-collected, discrete "bird dropping" RGB label — but no public download link,
  repository, or license has ever been located for it, in the original paper or afterward; it
  appears to be a private/internal dataset. Not accepted absent public access and a stated license.
- **Conclusion: still BLOCKED.** No source found that provides genuinely-provenanced, discretely-labeled
  Bird-Drop images under a verifiable license.

**Electrical-Damage / Physical-Damage — Zenodo record 18205662 re-confirmed live:**
- Fresh API check today: `access_right: restricted`, files array still empty, license still
  CC BY 4.0, owner/DOI unchanged (Hamdan Gani, Politeknik ATI Makassar,
  10.5281/zenodo.18205662). The record's own metadata additionally reports ~23,499 images across
  electroluminescence, thermal-infrared, and RGB modalities in the full (still-restricted) dataset.
- `DATA_ACCESS_REQUEST.md` status is still PENDING — no submission or approval evidence exists.
  Submitting it requires the repository owner's own Zenodo account/identity; it has not been
  submitted as of this re-verification.
- **Conclusion: still RESTRICTED.** No bypass attempted; no unauthorized access obtained.

## CORRECTION (2026-09-03)

Bird-Drop was previously marked resolved based on a license-compatibility review, but that review
assumed the DeepSolarEye dataset provided a discrete `bird-dropping` label. On actually downloading
and inspecting the full 45,755-file archive (`Solar_Panel_Soiling_Image_dataset/PanelImages/`), its
own README confirms every image is labeled only with continuous power-loss and irradiance values for
regression (`solar_..._L_<loss>_I_<irradiance>.jpg`) — there is no discrete soiling-type label, folder,
or metadata anywhere in the release. There is therefore no legitimate way to derive `Bird-Drop` ground
truth from this source without fabricating labels. **Bird-Drop reverts to NO USABLE SOURCE FOUND.**
See LICENSE_COMPATIBILITY.md for the full correction note.

Three classes (Clean, Dusty, Hotspot) have verified, accessible, clearly licensed sources.
Bird-Drop has no usable source (see correction above — the only public download for the previously
identified candidate lacks any discrete class label).
Two classes (Electrical-Damage, Physical-Damage) remain access-restricted on Zenodo pending
owner approval.

## Status Definitions

- **VERIFIED + USABLE**: License confirmed, data publicly downloadable, class mapping confirmed, all provenance requirements met
- **VERIFIED + LICENSE REVIEW REQUIRED**: License confirmed, data publicly downloadable, class mapping confirmed, but license has restrictions (e.g., NonCommercial) that require review against intended use
- **REQUEST REQUIRED**: License clear, but data access requires owner approval
- **RESTRICTED**: License clear, but access controls prevent download
- **REJECTED**: Candidate investigated and excluded for documented reasons
- **UNKNOWN**: License or provenance cannot be established

## Provenance Table

| Production Class | Dataset | Original Source | URL | License | Access | Original Labels | Mapping | Status |
|------------------|---------|-----------------|-----|---------|--------|-----------------|---------|---------|
| Clean | Solar Panel Dust Detection Dataset (SolNET) | MDPI Energies 2023 / Onimee58 | https://github.com/Onimee58/SolNET | CC BY 4.0 | VERIFIED (Google Drive) | clean | Direct | VERIFIED + USABLE |
| Dusty | Solar Panel Dust Detection Dataset (SolNET) | MDPI Energies 2023 / Onimee58 | https://github.com/Onimee58/SolNET | CC BY 4.0 | VERIFIED (Google Drive) | dirty/dusty | Direct | VERIFIED + USABLE |
| Hotspot | PVMD — Photovoltaic module dataset for automated fault detection | Tshwane University of Technology | https://data.mendeley.com/datasets/5ssmfpgrpc/1 | CC BY 4.0 | VERIFIED (Mendeley) | Hotspots | Direct | VERIFIED + USABLE |
| Bird-Drop | Solar Panel Soiling Image Dataset (SPSI/DeepSolarEye) | IBM Research / IEEE WACV 2018 | https://deep-solar-eye.github.io/ | CC BY-NC-SA 2.0 | Downloaded, inspected 2026-09-03 | none (regression-only, no class labels) | N/A — no discrete label exists | NO USABLE SOURCE FOUND |
| Electrical-Damage | Two-Stage Lightweight CNN RGB Dataset | Hamdan Gani / Politeknik ATI Makassar | https://doi.org/10.5281/zenodo.18205662 | CC BY 4.0 | RESTRICTED | Electrical Damage | Direct | RESTRICTED |
| Physical-Damage | Two-Stage Lightweight CNN RGB Dataset | Hamdan Gani / Politeknik ATI Makassar | https://doi.org/10.5281/zenodo.18205662 | CC BY 4.0 | RESTRICTED | Physical Damage | Direct | RESTRICTED |

## Detailed Source Records

### VERIFIED Sources

#### 1. Clean — Solar Panel Dust Detection Dataset (SolNET)
- **Dataset name**: Solar Panel Dust Detection Dataset
- **Original source/owner**: MDPI Energies 2023 — Onimee58 / SolNET authors
- **Publication**: "SolNet: A Convolutional Neural Network for Detecting Dust on Solar Panels" — MDPI Energies 2023, 16(1), 155
- **DOI**: https://doi.org/10.3390/en16010155
- **URL**: https://github.com/Onimee58/SolNET
- **Data download**: https://drive.google.com/drive/folders/12Q3MBI8SPw0vHsO_kkS5izkxw0F7tXx4
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original label**: `clean`
- **Production mapping**: Clean → Clean (direct)
- **Approx. image count**: 1,130 clean images (out of 2,231 total: 1,130 clean, 1,101 dirty)
- **Access verified**: Yes — direct Google Drive link in repository README
- **Provenance notes**: Dataset was collected from various regions in Bangladesh, ensuring different dust levels. Images manually sorted and labeled. Final version uses 227×227×3 RGB images.

#### 2. Dusty — Solar Panel Dust Detection Dataset (SolNET)
- **Dataset name**: Solar Panel Dust Detection Dataset
- **Original source/owner**: MDPI Energies 2023 — Onimee58 / SolNET authors
- **Publication**: "SolNet: A Convolutional Neural Network for Detecting Dust on Solar Panels" — MDPI Energies 2023, 16(1), 155
- **DOI**: https://doi.org/10.3390/en16010155
- **URL**: https://github.com/Onimee58/SolNET
- **Data download**: https://drive.google.com/drive/folders/12Q3MBI8SPw0vHsO_kkS5izkxw0F7tXx4
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original label**: `dirty` / `dusty`
- **Production mapping**: dirty → Dusty (direct)
- **Approx. image count**: 1,101 dirty images (out of 2,231 total)
- **Access verified**: Yes — direct Google Drive link in repository README
- **Provenance notes**: Same dataset as Clean class above. Binary classification (clean vs dirty) where dirty class includes dust and other contaminants.

#### 3. Hotspot — PVMD Dataset
- **Dataset name**: Photovoltaic module dataset for automated fault detection and analysis in large photovoltaic systems
- **Original source/owner**: Bello, Owolawi, Van Wyk, Du — Tshwane University of Technology, South Africa
- **Publication**: "Photovoltaic module dataset for automated fault detection and analysis in large photovoltaic systems" — Data in Brief, 2024
- **DOI**: https://doi.org/10.17632/5ssmfpgrpc.1 (Dataset); Paper: https://doi.org/10.1016/j.dib.2024.111184
- **URL**: https://data.mendeley.com/datasets/5ssmfpgrpc/1
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original label**: `Hotspots`
- **Production mapping**: Hotspots → Hotspot (direct)
- **Approx. image count**: 350 thermal images (Hotspots folder); 1,000 total (Hotspots: 350, Cracks: 350, Shadings: 300)
- **Access verified**: Yes — publicly downloadable from Mendeley Data
- **Provenance notes**: Thermal-infrared images captured with DJI Mavic 3 Thermal on September 5, 2024 at Soshanguve South Campus. Dataset explicitly designed for fault detection with supervised labels. Note: thermal modality only, not RGB.

#### 4. Bird-Drop — Solar Panel Soiling Image Dataset (SPSI/DeepSolarEye)
- **Dataset name**: Solar Panel Soiling Image Dataset (SPSI) / DeepSolarEye Dataset
- **Original source/owner**: IBM Research Lab — S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar, S. Kalyanaraman
- **Publication**: "DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels" — IEEE WACV 2018
- **URL**: https://deep-solar-eye.github.io/
- **Data download**: https://drive.google.com/open?id=1qB5dPWZMi2-12sLHDykHb9i6GibbJ46l
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 2.0 Generic (CC BY-NC-SA 2.0)
- **Original label**: `bird-dropping`
- **Production mapping**: NONE — see 2026-09-03 correction at top of this document. The full 45,755-file archive was downloaded and inspected; it contains one flat `PanelImages/` folder with images labeled only by continuous power-loss/irradiance values in the filename (per the archive's own README). No `bird-dropping` folder, tag, or metadata exists anywhere in the release.
- **Approx. image count**: 45,755 RGB images total, confirmed by full download and inspection 2026-09-03 — but none carry a discrete class label; the "6 soiling types" language in the paper is a description of the authors' own analysis categories, not a label present in the released data.
- **Access verified**: Yes — direct Google Drive link on project page; full archive downloaded and opened 2026-09-03.
- **Status**: NO USABLE SOURCE FOUND (reopened 2026-09-03). The earlier non-commercial license resolution is moot — it answered a licensing question for a dataset that turns out not to have the class labels this project needs at all. See LICENSE_COMPATIBILITY.md correction and DATA_ACCESS_REQUEST.md are not applicable here; a genuinely new labeled source must be identified.
- **Access notes**: Not applicable — the license/access question is moot until a labeled source exists.
- **Action required**: Identify and verify (by actual download and inspection, not paper/webpage description) a new Bird-Drop source with real discrete labels.

### REQUEST REQUIRED Sources

#### 5. Electrical-Damage — Two-Stage Lightweight CNN RGB Dataset
- **Dataset name**: Aerial RGB PV Dataset (part of multimodal Zenodo deposit)
- **Original source/owner**: Hamdan Gani — Politeknik ATI Makassar
- **Publication**: "Two-Stage Lightweight CNN with Weakly Supervised Defect Localization for Multimodal Photovoltaic Fault Detection" — Zenodo, 2026
- **DOI**: https://doi.org/10.5281/zenodo.18205662
- **URL**: https://zenodo.org/record/18205662
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original label**: `Electrical Damage`
- **Production mapping**: Electrical Damage → Electrical-Damage (direct)
- **Approx. image count**: ~145 images (875 total RGB images across 6 classes)
- **Access status**: RESTRICTED — Zenodo files array is empty; access_right: restricted
- **Access notes**: Data is NOT publicly downloadable. Zenodo access request workflow may be available. Contact author at Hamdan Gani, Politeknik ATI Makassar (email: verified institutional address at atim.ac.id per Google Scholar). Zenodo access requests can be submitted if the owner has enabled them.
- **Action required**: Submit access request via Zenodo or contact author directly to request dataset access for research purposes.

#### 6. Physical-Damage — Two-Stage Lightweight CNN RGB Dataset
- **Dataset name**: Aerial RGB PV Dataset (part of multimodal Zenodo deposit)
- **Original source/owner**: Hamdan Gani — Politeknik ATI Makassar
- **Publication**: "Two-Stage Lightweight CNN with Weakly Supervised Defect Localization for Multimodal Photovoltaic Fault Detection" — Zenodo, 2026
- **DOI**: https://doi.org/10.5281/zenodo.18205662
- **URL**: https://zenodo.org/record/18205662
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original label**: `Physical Damage`
- **Production mapping**: Physical Damage → Physical-Damage (direct)
- **Approx. image count**: ~145 images (875 total RGB images across 6 classes)
- **Access status**: RESTRICTED — Zenodo files array is empty; access_right: restricted
- **Access notes**: Same dataset as Electrical-Damage above. Same access restrictions apply.
- **Action required**: Submit access request via Zenodo or contact author directly.

### REJECTED Sources

| Dataset | Reason for Rejection |
|---------|---------------------|
| pythonafroz/solar-panel-images (Kaggle) | Unknown license — Kaggle page explicitly lists "License: Unknown". Peer-reviewed literature (SolarFCD) confirms license status as "permission uncertain". |
| pythonafroz/solar-panel-clean-and-faulty-images (Kaggle) | Unknown license — same as above, alternate Kaggle mirror. |
| Zenodo 18205662 — Thermal/EL subsets | Access restricted — files not publicly downloadable despite CC BY 4.0 license. |
| SolarFCD unified dataset | RGB-derived electrical-damage and physical-damage subsets inherit unknown license from Kaggle source; paper explicitly flags this. Thermal subsets are clean but use different class labels (hotspots/cracks). |
| InfraredSolarModules (RaptorMaps) | Thermal-only dataset; no RGB images. No explicit "electrical damage" or "physical damage" class. License stated in citing paper but not on repository. |
| PVF-10 | Thermal-only dataset. No explicit "electrical damage" or "physical damage" class. License not explicitly stated. |
| PVMD — Cracks/Shadings | While CC BY 4.0 and downloadable, class labels are `Cracks` and `Shadings`, not `Physical-Damage` or `Electrical-Damage`. Mapping would be post-hoc relabeling without source validation. |
| elpv-dataset | Does not explicitly define "physical damage" or "electrical damage" as discrete classes. Uses continuous defect-probability annotations. |
| All Roboflow object-detection datasets | Bounding-box annotations, not classification images. Excluded per criteria. |

## Missing Classes Status

| Class | Searches Performed | Sources Investigated | Best Candidate | License | Access Status | Reason Blocked | Author Contact Required |
|-------|-------------------|---------------------|----------------|---------|---------------|----------------||----------------------|
| Bird-Drop | Web search across arXiv, Mendeley, Zenodo, Roboflow, Kaggle, GitHub, IEEE, institutional repos, plus direct archive inspection 2026-09-03 | 15+ datasets | none — SPSI/DeepSolarEye ruled out, no other candidate found | N/A | N/A | Only candidate dataset has no discrete class labels (regression-only) | NO — needs a new source, not owner contact |
| Electrical-Damage | Web search across arXiv, Mendeley, Zenodo, GitHub, IEEE | 12+ datasets | Two-Stage Lightweight CNN RGB Dataset (Zenodo 18205662) | CC BY 4.0 | RESTRICTED | Access controls prevent download | YES |
| Physical-Damage | Web search across arXiv, Mendeley, Zenodo, GitHub, IEEE | 12+ datasets | Two-Stage Lightweight CNN RGB Dataset (Zenodo 18205662) | CC BY 4.0 | RESTRICTED | Access controls prevent download | YES |

## Recommended Next Action

**Two independent blockers must both clear: a new Bird-Drop source must be found, and the Zenodo access request for Electrical-Damage/Physical-Damage must be submitted and approved**

Rationale:
- Bird-Drop: BLOCKED (new reason as of 2026-09-03). The license question is moot — the only identified candidate dataset (SPSI/DeepSolarEye) was downloaded and inspected in full and contains no discrete class labels at all, only continuous power-loss/irradiance regression values. A genuinely new, labeled source must be found; none has been identified yet.
- Electrical-Damage and Physical-Damage: Only candidate with explicit class labels and permissive license (CC BY 4.0) is the Zenodo 18205662 RGB dataset, but it is access-restricted. Access request must be submitted to author Hamdan Gani (Politeknik ATI Makassar) via Zenodo or direct email — see DATA_ACCESS_REQUEST.md.

**Until access is granted for Electrical-Damage and Physical-Damage, six-class training cannot proceed.**

## Training Gate

**TRAINING STATUS = BLOCKED (Electrical-Damage / Physical-Damage access only)**

Training may begin only when ALL of the following conditions are satisfied:

- [x] Clean provenance verified
- [x] Dusty provenance verified
- [ ] Bird-Drop data source found (reopened 2026-09-03 — prior candidate has no usable labels; license question is moot until a labeled source exists)
- [ ] Electrical-Damage data legitimately accessible (Zenodo access request pending)
- [ ] Physical-Damage data legitimately accessible (Zenodo access request pending)
- [ ] All six class mappings verified
- [ ] Dataset preparation manifest generated
- [ ] Duplicate detection completed
- [ ] Train/validation/test split generated
- [ ] No prohibited data included

## Absolute Rule Compliance

- No fabricated data used
- No synthetic data presented as real
- No unknown-license datasets accepted
- No ambiguous label mappings accepted
- No Kaggle mirrors with unknown license accepted
- No object-detection bounding-box datasets accepted
- No reduction of six-class production contract
- No training performed without all six classes verified

## Current Blocker Summary

1. **Electrical-Damage / Physical-Damage**: Access-restricted on Zenodo. Data access request must be submitted to Hamdan Gani (Politeknik ATI Makassar). See DATA_ACCESS_REQUEST.md.
2. **Bird-Drop**: BLOCKED (reopened 2026-09-03) — the only identified candidate dataset (SPSI/DeepSolarEye) was downloaded and inspected in full; it has no discrete class labels (regression-only). See LICENSE_COMPATIBILITY.md correction note. A new, genuinely labeled source is needed.

**NO TRAINING PERFORMED — PROVENANCE/ACCESS GATE NOT YET SATISFIED (two independent blockers: Bird-Drop has no labeled source, Electrical-Damage/Physical-Damage remain Zenodo-restricted).**
