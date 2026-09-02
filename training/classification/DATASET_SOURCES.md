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

Three classes (Clean, Dusty, Hotspot) have verified, accessible, clearly licensed sources.
Bird-Drop has a verified, accessible source; its NonCommercial license restriction was resolved
on 2026-09-03 based on the project's current non-commercial intent (see LICENSE_COMPATIBILITY.md).
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
| Bird-Drop | Solar Panel Soiling Image Dataset (SPSI/DeepSolarEye) | IBM Research / IEEE WACV 2018 | https://deep-solar-eye.github.io/ | CC BY-NC-SA 2.0 | VERIFIED (Google Drive) | bird-dropping | Direct | VERIFIED + USABLE (non-commercial) |
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
- **Production mapping**: bird-dropping → Bird-Drop (direct)
- **Approx. image count**: 45,754 RGB images total; bird-dropping is one of 6 soiling types
- **Access verified**: Yes — direct Google Drive link on project page
- **Status**: VERIFIED + USABLE (non-commercial) — CC BY-NC-SA 2.0 imposes a NonCommercial restriction. Project maintainer confirmed on 2026-09-03 that Solar AI is research/non-commercial at this time, so this class is cleared for use under that intent. See LICENSE_COMPATIBILITY.md for full analysis.
- **Access notes**: Data is publicly downloadable via Google Drive. Attribution to IBM Research / DeepSolarEye authors must be preserved in documentation and any model card.
- **Action required**: None under current non-commercial intent. If commercial deployment is later planned, this determination must be revisited and owner permission requested (see LICENSE_COMPATIBILITY.md).

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
| Bird-Drop | Web search across arXiv, Mendeley, Zenodo, Roboflow, Kaggle, GitHub, IEEE, institutional repos | 15+ datasets | SPSI/DeepSolarEye | CC BY-NC-SA 2.0 | VERIFIED | RESOLVED 2026-09-03 — non-commercial intent confirmed, cleared for use | NO |
| Electrical-Damage | Web search across arXiv, Mendeley, Zenodo, GitHub, IEEE | 12+ datasets | Two-Stage Lightweight CNN RGB Dataset (Zenodo 18205662) | CC BY 4.0 | RESTRICTED | Access controls prevent download | YES |
| Physical-Damage | Web search across arXiv, Mendeley, Zenodo, GitHub, IEEE | 12+ datasets | Two-Stage Lightweight CNN RGB Dataset (Zenodo 18205662) | CC BY 4.0 | RESTRICTED | Access controls prevent download | YES |

## Recommended Next Action

**Electrical-Damage / Physical-Damage access request must be submitted and approved**

Rationale:
- Bird-Drop: RESOLVED. Project maintainer confirmed on 2026-09-03 that Solar AI is research/non-commercial at this time, clearing the CC BY-NC-SA 2.0 NonCommercial restriction under that intent. This must be revisited if commercial deployment is later planned (see LICENSE_COMPATIBILITY.md).
- Electrical-Damage and Physical-Damage: Only candidate with explicit class labels and permissive license (CC BY 4.0) is the Zenodo 18205662 RGB dataset, but it is access-restricted. Access request must be submitted to author Hamdan Gani (Politeknik ATI Makassar) via Zenodo or direct email — see DATA_ACCESS_REQUEST.md.

**Until access is granted for Electrical-Damage and Physical-Damage, six-class training cannot proceed.**

## Training Gate

**TRAINING STATUS = BLOCKED (Electrical-Damage / Physical-Damage access only)**

Training may begin only when ALL of the following conditions are satisfied:

- [x] Clean provenance verified
- [x] Dusty provenance verified
- [x] Bird-Drop license/use approved (non-commercial intent confirmed 2026-09-03; revisit before commercial deployment)
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
2. **Bird-Drop**: RESOLVED 2026-09-03 — non-commercial intent confirmed by project maintainer, CC BY-NC-SA 2.0 compatible under that intent. See LICENSE_COMPATIBILITY.md. Must be revisited before any commercial deployment.

**NO TRAINING PERFORMED — PROVENANCE/ACCESS GATE NOT YET SATISFIED (Zenodo access remains the sole blocker).**
