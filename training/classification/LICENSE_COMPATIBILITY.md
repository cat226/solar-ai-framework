# License Compatibility Review — Bird-Drop Source

Generated: 2026-08-14
Repository: https://github.com/cat226/solar-ai-framework
Branch: feat/mobilenet-training-pipeline

## Source Under Review

- **Dataset**: Solar Panel Soiling Image Dataset (SPSI) / DeepSolarEye Dataset
- **Original source/owner**: IBM Research Lab — S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar, S. Kalyanaraman
- **Publication**: "DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels" — IEEE WACV 2018
- **URL**: https://deep-solar-eye.github.io/
- **Data download**: https://drive.google.com/open?id=1qB5dPWZMi2-12sLHDykHb9i6GibbJ46l
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 2.0 Generic (CC BY-NC-SA 2.0)
- **Original label**: `bird-dropping`
- **Production mapping**: bird-dropping → Bird-Drop (direct)

## CC BY-NC-SA 2.0 Analysis

### What CC BY-NC-SA 2.0 Permits

CC BY-NC-SA 2.0 allows:
- Sharing — copy and redistribute the material in any medium or format
- Adaptation — remix, transform, and build upon the material

### Conditions

1. **Attribution**: You must give appropriate credit to the original author(s), provide a link to the license, and indicate if changes were made.
2. **NonCommercial**: You may not use the material for commercial purposes.
3. **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

### Key Restrictions for This Project

- **NonCommercial (NC)**: The dataset may not be used for commercial purposes. This means:
  - Training a model on this data for commercial deployment is prohibited without additional permission
  - Distributing a model trained on this data for commercial use is prohibited
  - Using this data in a commercial product or service is prohibited

- **ShareAlike (SA)**: Any derivative works must be licensed under CC BY-NC-SA 2.0. This raises questions about:
  - Model weights trained on this data: Are model weights a "derivative work" under CC BY-NC-SA 2.0?
  - This is a legally ambiguous area. Some interpretations suggest model weights are functional objects, not derivative works in the copyright sense. Others argue they are derivative because they contain extracted knowledge from the training data.

### Downstream Licensing Concerns

1. **Model weights**: If model weights are considered a derivative work, they would need to be released under CC BY-NC-SA 2.0, which would prohibit commercial use of the model.
2. **Combined datasets**: If this dataset is combined with other datasets (e.g., SolNET CC BY 4.0), the resulting work may need to comply with the most restrictive license.
3. **Production deployment**: Commercial deployment of a model trained on this data would require:
   - Explicit permission from the copyright holder, OR
   - A legal determination that model weights are not derivative works

## Repository Commercial Intent Assessment

The repository does not explicitly state commercial or non-commercial intent. The production classifier is intended for:
- Solar panel condition monitoring
- Maintenance optimization
- Energy production efficiency

Without explicit commercial/non-commercial designation from the repository owner, the safest assumption is that production deployment may have commercial implications.

## Compatibility Status

**COMPATIBLE — NON-COMMERCIAL USE CONFIRMED (2026-09-03)**

The project maintainer confirmed on 2026-09-03 that Solar AI is being developed for research / non-commercial use at this time. Under that intent, CC BY-NC-SA 2.0 is compatible: the dataset may be used for training and evaluation subject to the Attribution and ShareAlike conditions below.

This determination is scoped to the current, stated intent. **It must be revisited before any commercial deployment** — see Recommendations.

Rationale:
- CC BY-NC-SA 2.0 permits non-commercial use, sharing, and adaptation with attribution and ShareAlike
- Repository maintainer has explicitly confirmed non-commercial intent (see Action Required, resolved below)
- Attribution to IBM Research / DeepSolarEye authors must be preserved in documentation and any model card

## Recommendations

1. **Current status (research-only / non-commercial)**: CC BY-NC-SA 2.0 is compatible. Proceed with attribution and ShareAlike compliance — maintain credit to IBM Research / DeepSolarEye authors in documentation and model cards.
2. **If commercial deployment is later intended**: This determination does not carry over automatically. Before any commercial use, contact the dataset owners (IBM Research / DeepSolarEye authors) to request:
   - Explicit permission for commercial use, OR
   - A license upgrade to a more permissive license (e.g., CC BY 4.0)
   - Alternatively, source a CC BY-licensed replacement for the Bird-Drop class
3. **Regardless of intent**: Maintain clear attribution to the original dataset and authors in all documentation and model cards.

## Action Required

**RESOLVED (2026-09-03)**: Project maintainer confirmed the Solar AI project is research / non-commercial at this time. No further owner permission is needed under the current intent. If commercial deployment is later planned, this compatibility review must be reopened.

## Absolute Rule Compliance

- No fabricated licensing status
- No assumption of commercial intent
- No conclusion beyond what the license text supports
- Flagged for human/legal review as required
