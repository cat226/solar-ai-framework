# Condition dataset — overlap report

Signals: exact SHA-256, 16x16 dHash (Hamming <= 6), source basename, dimensions.

## vs the existing 6,492-image leakage-clean ground detector dataset (`datasets/ground_pv_research/yolo/`)

- existing images: 6492
- condition-dataset images: 2509
- exact SHA matches: 0
- perceptual near-duplicate images: 33
- overlap fraction: 0.0132

## within S1 + S3 (dedup families)

- total source images: 2673
- unique after dedup: 2509
- near-duplicate families: 1244
- exact byte-duplicates removed within the pool: 6
