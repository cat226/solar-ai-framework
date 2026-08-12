# Cloud Attenuation Model

The physics layer treats OpenWeatherMap cloud cover as a direct attenuation signal.

- 0% cloud cover -> factor 1.0
- 50% cloud cover -> factor 0.5
- 90% cloud cover -> factor 0.1
- 100% cloud cover -> factor 0.0

Inputs are clamped to 0–100% before conversion. The resulting factor is applied
multiplicatively to the clear-sky irradiance estimate.

This is intentionally deterministic and keeps the existing solar-geometry and
NOCT calculations unchanged. The model does not fabricate irradiance data when
weather or model artifacts are unavailable.
