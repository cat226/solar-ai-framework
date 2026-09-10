# Active 4-class condition dataset (`condition_4class_v1`) — build report

**STATUS: APPROVED**

- derived from (read-only, verified unchanged: True): `D:\Documents\Projects\solar-ai-framework\datasets\ground_pv_research\condition`
- output: `D:\Documents\Projects\solar-ai-framework\datasets\ground_pv_research\condition_4class_v1`

## Removal effect

- source images (5-class): 2509
- derived images (4-class): 2411
- ELECTRICAL_DAMAGE instances removed (never relabelled): 1119
- images dropped (ALL boxes were ELECTRICAL_DAMAGE): 98
- images kept with ELECTRICAL_DAMAGE partially removed (other classes remain): 381

## Split counts (preserved exactly from the 5-class leakage-clean split)

| split | images |
|---|---|
| train | 1686 |
| valid | 483 |
| test | 242 |

## Class distribution (4-class)

| class | train img | valid img | test img | total img | total instances |
|---|---|---|---|---|---|
| CLEAN | 639 | 184 | 95 | 918 | 6000 |
| DUSTY | 959 | 261 | 144 | 1364 | 8550 |
| BIRD_DROP | 737 | 194 | 100 | 1031 | 7742 |
| PHYSICAL_DAMAGE | 205 | 79 | 30 | 314 | 725 |

## Quality gates

- electrical_damage_never_relabelled: True
- no_defective_class_present: True
- no_electrical_damage_class_present: True
- no_exact_duplicate_crosses_splits: True
- no_family_crosses_splits: True
- no_panel_class_present: True
- no_snow_class_present: True
- **ALL PASS: True**

