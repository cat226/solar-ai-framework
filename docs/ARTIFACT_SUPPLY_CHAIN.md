# Model artifact supply chain

The repository does not contain trained model weights. Deployment artifacts are supplied separately and must be verified before inference is enabled.

## Integrity verification

Create `weights/manifest.json` in the deployment artifact bundle. Each entry must contain a relative `path` and a 64-character SHA-256 digest:

```json
{
  "artifacts": [
    {"path": "yolo_solar.pt", "sha256": "<reviewed digest>"},
    {"path": "mobilenet_solar.pth", "sha256": "<reviewed digest>"},
    {"path": "xgboost_solar.joblib", "sha256": "<reviewed digest>"}
  ]
}
```

Run:

```text
python scripts/verify_model_artifacts.py --manifest weights/manifest.json
```

The verifier only reads the manifest and existing files. It never downloads, creates, or substitutes artifacts. Paths are confined to the manifest directory to prevent a manifest from directing verification outside the supplied artifact bundle.

## Promotion requirements

Before accepting a model bundle for production:

1. Obtain the artifacts from the controlled model-training/release process.
2. Review the artifact provenance and intended model/version identifiers.
3. Generate SHA-256 digests from the exact files being deployed.
4. Review the manifest independently of the artifact producer.
5. Run the verifier successfully in the deployment environment.
6. Run the runtime readiness check and confirm all required artifacts are present.
7. Record the artifact source and digest set in the deployment record.

Do not commit fabricated weights or placeholder binaries to make readiness pass.
