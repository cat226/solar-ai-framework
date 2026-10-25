"""scripts/label_panel_conditions.py — owner's per-panel condition labeling tool.

Run (from the repo root):
    .venv\\Scripts\\streamlit.exe run scripts/label_panel_conditions.py -- ^
        --images "F:\\path\\to\\pristine\\photos" --out "F:\\Solar AI Derived\\owner_labels\\owner_condition_labels.csv"

Shows each photo with the detector's numbered panel boxes (proposals), one crop per
panel, and records YOUR label per panel: CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE /
SKIP (unsure, or not a panel). Resumable (already-labeled photos are skipped). A photo can
only be saved once every panel has a label, so a skipped click never reads as "clean".

Point --images at PRISTINE photos (no drawn boxes/text on them): marks burned into a photo
would leak the label into training. Tick "staged/simulated defect" for any photo whose
defect you created on purpose, so real and simulated examples stay separable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv import owner_condition_labels as ocl  # noqa: E402

_DEFAULT_OUT = _REPO_ROOT / "datasets" / "ground_pv_research" / "owner_labels" / "owner_condition_labels.csv"
_MAX_DISPLAY_PX = 1400


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.environ.get("LABEL_TOOL_IMAGES", ""))
    ap.add_argument("--out", default=os.environ.get("LABEL_TOOL_OUT", str(_DEFAULT_OUT)))
    ap.add_argument("--conf", type=float, default=0.25)
    ns, _ = ap.parse_known_args(sys.argv[1:])
    return ns


@st.cache_resource(show_spinner="Loading panel detector...")
def _panel_model():
    from services.ground_pipeline import PANEL_WEIGHTS
    from ultralytics import YOLO

    return YOLO(str(PANEL_WEIGHTS))


def _draw(image: Image.Image, panels: list[dict]) -> Image.Image:
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(3, im.width // 400)
    for i, p in enumerate(panels):
        b = p["bbox"]
        d.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(255, 215, 0), width=lw)
        d.rectangle([b["x1"], b["y1"], b["x1"] + 14 * lw, b["y1"] + 9 * lw], fill=(255, 215, 0))
        d.text((b["x1"] + lw, b["y1"] + lw), str(i + 1), fill=(0, 0, 0))
    return im


def main() -> None:
    st.set_page_config(page_title="Panel condition labeling", layout="wide")
    a = _args()
    st.title("Panel condition labeling")
    if not a.images:
        st.error("Start with:  streamlit run scripts/label_panel_conditions.py -- --images <folder of pristine photos>")
        return
    out_csv = Path(a.out)
    try:
        images = ocl.list_images(Path(a.images))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    done = ocl.labeled_image_keys(out_csv)
    todo = [p for p in images if (p.name, ocl.sha256_file(p)) not in done]
    summ = ocl.summarize(out_csv)
    st.sidebar.metric("Photos labeled", f"{summ['images']} / {len(images)}")
    st.sidebar.write({k: v for k, v in summ["by_label"].items()})
    st.sidebar.caption(f"simulated panels: {summ['simulated_panels']}  |  output: {out_csv}")
    conf = st.sidebar.slider("Detector confidence", 0.05, 0.9, a.conf, 0.05)

    if not todo:
        st.success("All photos in this folder are labeled.")
        return
    path = st.sidebar.selectbox("Photo", todo, format_func=lambda p: p.name)
    image = Image.open(path)
    image.load()
    sha = ocl.sha256_file(path)

    panels = ocl.propose_panels(_panel_model(), image, image_name=path.name, conf=conf)
    shown = _draw(image, panels)
    shown.thumbnail((_MAX_DISPLAY_PX, _MAX_DISPLAY_PX))
    st.image(shown, caption=f"{path.name}: {len(panels)} proposed panel(s). Boxes are the detector's proposals.")
    if not panels:
        st.warning("No panels proposed. Lower the confidence slider, or record the count as missed below and save.")

    key = f"{path.name}:{sha[:8]}"
    cols = st.columns(min(4, max(1, len(panels))))
    chosen: list[str | None] = []
    for i, p in enumerate(panels):
        b = p["bbox"]
        crop = image.convert("RGB").crop((int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])))
        crop.thumbnail((360, 360))
        with cols[i % len(cols)]:
            st.image(crop, caption=f"Panel {i + 1}")
            chosen.append(st.radio("Condition", ocl.LABELS, index=None, key=f"{key}:{i}", label_visibility="collapsed"))

    if panels and st.button("Mark every panel CLEAN (no defect anywhere)"):
        for i in range(len(panels)):
            st.session_state[f"{key}:{i}"] = "CLEAN"
        st.rerun()

    missed = st.number_input("Panels the detector MISSED in this photo", min_value=0, value=0, step=1, key=f"{key}:missed")
    simulated = st.checkbox("Staged / simulated defect in this photo", key=f"{key}:sim")
    notes = st.text_input("Notes (optional)", key=f"{key}:notes")

    ready = all(c is not None for c in chosen)
    if st.button("Save and next", type="primary", disabled=not ready):
        rows = ocl.build_rows(path.name, sha, panels, [c for c in chosen if c], missed_panels=int(missed),
                              simulated=simulated, notes=notes)
        ocl.append_rows(out_csv, rows)
        st.rerun()
    if not ready:
        st.caption("Label every panel to enable saving.")


main()
