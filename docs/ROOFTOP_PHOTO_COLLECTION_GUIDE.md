# Rooftop photo collection guide (stand-in for the Raspberry Pi camera)

Why: the web-photo pool does not transfer to this rooftop (see
`datasets/ground_pv_research/unified_condition_v1/README.md`). The limiting factor is
**independent defect scenes** from the real camera position, not image count.

## What to capture
1. **One fixed spot**: same position, height, angle and zoom as the future Pi camera. Do not
   re-frame between shots. Use the phone's main lens, no digital zoom, no filters/HDR beauty modes.
2. **Over time**: at least 10 different days, and at different times (morning / noon / evening /
   after rain / after a windy day). Same scene on different days is what the model must handle.
3. **Natural conditions only**: photograph what happens (dust build-up, bird drops, rain
   streaks, any damage you notice). Do not stage or create damage on panels.
4. **A few shots per visit** (3-5, a second apart) is enough; near-identical frames add little.
5. **Close-ups for ground truth**: when you see a defect, also take 1-2 close photos of it from
   the same side so it can be confirmed as bird drop / dust / physical damage.

## Keep clean
- Do not draw boxes or text on the originals. Marks go in a separate copy (they leak labels).
- Keep original files (no WhatsApp re-compression if you can copy the phone files over a cable
  or cloud drive instead: WhatsApp shrinks a 4032 px photo to ~1280 px and erases small specks).

## Import and label
```
.venv\Scripts\python.exe scripts\import_first_party_panels.py --source "<new photos dir>" ^
    --batch-id batch_003 --scene-id <short_scene_name> --camera-id phone_fixed_spot
.venv\Scripts\streamlit.exe run scripts\label_panel_conditions.py -- ^
    --images "<new photos dir>" --out "F:\Solar AI Datasets\owner_labels\owner_condition_labels.csv"
```
The labeller records YOUR label per panel (CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE / SKIP).

## Review the AI-proposed labels for the existing 48 photos
Contact sheets: `F:\Solar AI Datasets\condition_unified_v1\review_sheets\review_01.jpg ...`
(regenerate with `scripts/make_label_review_sheets.py`). Correct the `label` column in
`rooftop48_ai_proposed_labels.csv`, or send the indices that are wrong.
