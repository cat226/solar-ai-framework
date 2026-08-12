"""Verification script for TASK-001R import checks."""
import sys
sys.path.insert(0, '.')

errors = []
passes = []

tests = [
    ('utils.exceptions', "from utils.exceptions import SolarAIError, ModelLoadError, PredictionError, ImageValidationError, FeatureValidationError, WeatherAPIError\nprint('  SolarAIError bases:', SolarAIError.__bases__)"),
    ('utils.config — CFG + get_secret', "from utils.config import CFG, get_secret\nprint('  CFG keys:', list(CFG.keys()))\nassert 'api_key' not in CFG.get('weather', {}), 'api_key still in YAML!'\ns = get_secret('NONEXISTENT_KEY', 'default')\nassert s == 'default', f'expected default, got {s}'\nprint('  get_secret fallback OK')"),
    ('utils.logger', "from utils.logger import get_logger\nlog = get_logger('test')\nlog.info('logger OK')"),
    ('utils.image_utils', "import utils.image_utils\nprint('  image_utils OK')"),
    ('models.model_manager', "from models.model_manager import ModelManager, model_manager\nprint('  loaded_models:', model_manager.loaded_models)"),
    ('models.detector', "from models.detector import DetectionResult, SolarPanelDetector\nd = SolarPanelDetector()\nprint('  detector created')"),
    ('models.classifier', "from models.classifier import ClassificationResult, SolarFaultClassifier\nc = SolarFaultClassifier()\nprint('  classifier created')"),
    ('models.predictor', "from models.predictor import PredictionResult, EnergyPredictor\np = EnergyPredictor()\nprint('  predictor created')"),
    ('services.weather', "from services.weather import WeatherData, fetch_weather\nprint('  weather imports OK')"),
    ('services.physics — compute_physics live call', "from services.physics import compute_physics, PhysicsResult\nr = compute_physics(25, 2, 30, 'Clean')\nprint(f'  irr={r.irradiance_wm2:.1f} W/m2, mod_temp={r.module_temp_c:.1f}C')"),
    ('services.feature_engineering — build_features + validate_features', "from services.feature_engineering import build_features, validate_features, build_feature_dataframe\nprint('  feature_engineering imports OK')"),
    ('services.recommendation — to_dict', "from services.recommendation import generate_recommendations, RecommendationReport, Recommendation, Severity\nr = RecommendationReport()\nd = r.to_dict()\nassert 'status' in d and 'issues' in d and 'recommendation' in d and 'priority' in d\nprint('  to_dict keys:', list(d.keys()))"),
    ('services.pipeline — run_pipeline + PipelineResult', "from services.pipeline import run_pipeline, PipelineResult\nprint('  pipeline imports OK')"),
    ('app.py — top-level parseable', "import ast\nwith open('app.py', encoding='utf-8') as f: src = f.read()\nast.parse(src)\nlines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith('#')]\nprint(f'  app.py parses OK, executable lines: {len(lines)}')"),
]

for name, code in tests:
    try:
        exec(code)
        passes.append(name)
        print(f'[PASS] {name}')
    except ImportError as e:
        errors.append((name, f'ImportError: {e}'))
        print(f'[FAIL] {name} — missing dependency: {e}')
    except AssertionError as e:
        errors.append((name, f'AssertionError: {e}'))
        print(f'[FAIL] {name} — assertion: {e}')
    except Exception as e:
        errors.append((name, str(e)))
        print(f'[FAIL] {name} — {e}')

print()
print(f'Results: {len(passes)}/{len(tests)} passed, {len(errors)} failed')
if errors:
    print('Failures:')
    for n, e in errors:
        print(f'  [{n}]: {e}')
    sys.exit(1)

sys.exit(0)
