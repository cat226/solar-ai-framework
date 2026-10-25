# Ground Inspection - current status and known limits (2026-10-26)

Read this before trusting any condition output.

## What exists
* **Panel finder** (`yolo_solar_rooftop_v1.pt`): found a panel under every one of the 7 owner-marked defects in the
  rooftop check. It can still miss or merge panels.
* **Condition classifier, default** (`rooftop_mix` MobileNetV2): its "0.938 on the rooftop's own photos" came from
  AI-assigned labels that the owner's own defect marks later contradicted. In a direct check it called **all 5**
  owner-marked bird-drop panels CLEAN at 0.96-1.00 confidence. **A CLEAN result does not mean the panel is clean.**
* **Condition classifier, web-trained ensemble** (opt-in, `web_ensemble` / `web_ensemble_lite`): the best the three
  public datasets allow. 5-fold photo-family-grouped out-of-fold: accuracy 0.899, macro-F1 0.878 (5 models) /
  0.896, 0.875 (2 models). Per class recall (5 models): CLEAN 0.94, DUSTY 0.84, BIRD_DROP 0.82 [0.74, 0.88],
  PHYSICAL_DAMAGE 0.85 [0.73, 0.93]. Details: `datasets/ground_pv_research/web_condition_v2/README.md`.
* **Alerts**: opt-in webhook/email on confident non-CLEAN reads (`services/notifications.py`), cooldown, always labelled
  unvalidated.

## What those numbers are NOT
* Not rooftop accuracy. They are web photos (stock images, ads, other people's roofs) from the same three sources the
  models learned from. A web-trained model once scored 79% on its web test and 0/7 on the owner's marked rooftop defects.
* Not independent of model selection: five configurations were compared on the same out-of-fold scores.
* Not free of label noise: about 8 in 10 of the remaining errors (212 of 263) are CLEAN vs DUSTY, which the source labels cannot settle.

## Known gaps
1. No labelled photos from the real fixed camera exist (the owner's earlier photos were deleted on request). Accuracy
   on the deployment camera is **unmeasured**.
2. Three source datasets have unresolved licences; research only, do not redistribute.
3. Rooftop-adapted weights (`yolo_solar_rooftop_v1.pt`, `rooftop_mix`) were learned from the deleted photos.
4. ELECTRICAL_DAMAGE and SNOW are deliberately outside the 4-class taxonomy.
5. Pi runtime of the ensembles is estimated, not measured.

## What would change this
About 30 original (not WhatsApp) photos from the fixed spot over 3-4 days, including some real bird drops or dust,
labelled by the owner: enough for a first real accuracy figure and a fine-tune. Protocol:
`docs/ROOFTOP_PHOTO_COLLECTION_GUIDE.md`.
