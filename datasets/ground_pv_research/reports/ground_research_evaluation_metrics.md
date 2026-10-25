# Ground-level PV research detector — evaluation

Weights: `weights\yolo_solar_ground_research_v1.pt`
SHA-256: `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50`
Ultralytics: 8.4.118  |  imgsz=640  iou=0.5

## Split: val

- images: 1297
- instances per class: {'panel': 1214, 'cracked': 2122, 'dusty': 75}
- images containing class: {'panel': 932, 'cracked': 832, 'dusty': 21}

**Overall**: mAP50=0.8562741882961172  mAP50-95=0.6629495477651525  P=0.8464937517538492  R=0.8088012798400426

| class | precision | recall | mAP50 | mAP50-95 | rare? |
|---|---|---|---|---|---|
| panel | 0.8959861966792585 | 0.8747940691927513 | 0.9270815461208219 | 0.8117401425444151 |  |
| cracked | 0.857254270706292 | 0.7782764369940431 | 0.8395425689430034 | 0.595584947085596 |  |
| dusty | 0.7862407878759973 | 0.7733333333333333 | 0.8021984498245263 | 0.5815235536654464 | **YES** |

## Split: test

- images: 649
- instances per class: {'panel': 560, 'cracked': 637, 'dusty': 69}
- images containing class: {'panel': 539, 'cracked': 328, 'dusty': 30}

**Overall**: mAP50=0.8647350484282449  mAP50-95=0.677469701852825  P=0.8755931641449556  R=0.801382727003845

| class | precision | recall | mAP50 | mAP50-95 | rare? |
|---|---|---|---|---|---|
| panel | 0.9465348599484664 | 0.925 | 0.9628127518464877 | 0.8427276123704651 |  |
| cracked | 0.8090230128673809 | 0.6530612244897959 | 0.7495067321869836 | 0.5295809416922473 |  |
| dusty | 0.8712216196190193 | 0.8260869565217391 | 0.8818856612512631 | 0.6601005514957624 | **YES** |

> `dusty` (and, in the 4-class taxonomy, `bird_drop`) are severely underrepresented; their per-class metrics are high-variance and must not be over-interpreted.

