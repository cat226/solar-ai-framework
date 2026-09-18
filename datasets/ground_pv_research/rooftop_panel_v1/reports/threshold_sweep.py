import sys, json
sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def labels(p):
    rows = [[float(v) for v in l.split()[1:5]] for l in p.read_text().splitlines() if l.strip()]
    a = np.array(rows).reshape(-1,4)
    return np.c_[a[:,0]-a[:,2]/2, a[:,1]-a[:,3]/2, a[:,0]+a[:,2]/2, a[:,1]+a[:,3]/2]

def iou(a,b):
    ix = np.maximum(0, np.minimum(a[2],b[:,2])-np.maximum(a[0],b[:,0]))
    iy = np.maximum(0, np.minimum(a[3],b[:,3])-np.maximum(a[1],b[:,1]))
    inter = ix*iy
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter+1e-9)

model = YOLO('weights/yolo_solar_rooftop_v1.pt')
ds = Path('datasets/ground_pv_research/rooftop_panel_v1')
imgs = sorted((ds/'images/test').glob('*.jpg'))
per_image = []
for im in imgs:
    gt = labels(ds/'labels/test'/f'{im.stem}.txt')
    r = model.predict(str(im), conf=0.05, iou=0.6, imgsz=640, verbose=False)[0]
    xy = r.boxes.xyxyn.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    per_image.append((im.name, gt, xy, cf))

for thr in [0.05,0.10,0.15,0.20,0.25,0.30,0.40]:
    tp=fp=0; fn=0; miss_detail=[]
    for name, gt, xy, cf in per_image:
        m = cf>=thr
        xy_,cf_ = xy[m], cf[m]
        used=set(); img_tp=0
        for i in np.argsort(-cf_):
            if len(gt)==0: break
            ious = iou(xy_[i], gt)
            j = np.argmax(ious)
            if ious[j]>=0.5 and j not in used:
                used.add(j); img_tp+=1
        tp += img_tp; fp += len(cf_)-img_tp; fn += len(gt)-img_tp
        if len(gt)-img_tp>0: miss_detail.append((name, len(gt)-img_tp))
    recall = tp/(tp+fn); prec = tp/(tp+fp) if tp+fp>0 else 0
    print(f"conf>={thr:.2f}: recall={recall:.3f} ({tp}/{tp+fn})  precision={prec:.3f}  fp={fp}")
    if thr==0.25:
        print("   missed per image:", miss_detail)
