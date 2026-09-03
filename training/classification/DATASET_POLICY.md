# Dataset provenance policy

Training data must be acquired externally and must not be committed to this repository.

For every source, retain a local record containing:

- source URL and dataset/version identifier
- access/download date
- license and attribution requirements
- original class names
- mapping into the six production classes
- duplicate/removal notes
- train/validation/test split assignment

The production classes are exactly:
`Clean`, `Dusty`, `Bird-Drop`, `Electrical-Damage`, `Physical-Damage`, `Hotspot`.

Never convert `Snow-Covered` into `Hotspot`, and never create synthetic samples and present them as real observations. A source without a clearly established license or provenance must not enter the training set.
