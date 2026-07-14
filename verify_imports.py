"""Verification script for TASK-001R import checks."""
import sys
sys.path.insert(0, '.')

errors = []
passes = []

tests = [
    (
        'utils.exceptions',
        "from utils.exceptions import SolarAIError, ModelLoadError, PredictionError, "
        "ImageValidationError, FeatureValidationError, WeatherAPIError\n"
        "print('  SolarAIError bases:', SolarAIError.__bases__)"
    ),
    (
        'utils.config — CFG + get_secret',
        "from utils.config import CFG, get_secret\n"
        "print('  CFG keys:', list(CFG.keys()))\n"
        "assert 'api_key' not in CFG.get('weather', {}), 'api_key still in YAML!'\n"
        "s = get_secret('NONEXISTENT_KEY', 'default')\n"
        "assert s == 'default', f'expected default, got {s}'\n"
        "print('  get_secret fallback OK')"
    ),
    (
        'utils.logger',
        "from utils.logger import get_logger\n"
        "log = get_logger('test')\n"
        "log.info('logger OK')"
    ),
    (
        'utils.image_utils',
        "import utils.image_utils\n"
        "print('  image_utils OK')"
    ),
    (
        'models.model_manager',
        "from models.model_manager import ModelManager, model_manager\n"
        "print('  loaded_models:', model_manager.loaded_models)"
    ),
    (
        'models.detector',
        "from models.detector import DetectionResult, SolarPanelDetector\n"
        "d = SolarPanelDetector()\n"
        "print('  detector created')"
    ),
    (
        'models.classifier',
        "from models.classifier import ClassificationResult, SolarFaultClassifier\n"
        "c = SolarFaultClassifier()\n"
        "print('  classifier created')"
    ),
    (
        'models.predictor',
        "from models.predictor import PredictionResult, EnergyPredictor\n"
        "p = EnergyPredictor()\n"
        "print('  predictor created')"
    ),
    (
        'services.weather',
        "from services.weather import WeatherData, fetch_weather\n"
        "print('  weather imports OK')"
    ),
    (
        'services.physics — compute_physics live call',
        "from services.physics import compute_physics, PhysicsResult\n"
        "r = compute_physics(25, 2, 30, 'Clean')\n"
        "print(f'  irr={r.irradiance_wm2:.1f} W/m2, mod_temp={r.module_temp_c:.1f}C')"
    ),
    (
        'services.feature_engineering — build_features + validate_features',
        "from services.feature_engineering import build_features, validate_features, build_feature_dataframe\n"
        "print('  feature_engineering imports OK')"
    ),
    (
        'services.recommendation — to_dict',
        "from services.recommendation import generate_recommendations, RecommendationReport, Recommendation, Severity\n"
        "r = RecommendationReport()\n"
        "d = r.to_dict()\n"
        "assert 'status' in d and 'issues' in d and 'recommendation' in d and 'priority' in d\n"
        "print('  to_dict keys:', list(d.keys()))"
    ),
    (
        'services.pipeline — run_pipeline + PipelineResult',
        "from services.pipeline import run_pipeline, PipelineResult\n"
        "print('  pipeline imports OK')"
    ),
    (
        'app.py — top-level parseable',
        "import ast\n"
        "with open('app.py', encoding='utf-8') as f: src = f.read()\n"
        "ast.parse(src)\n"
        "lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith('#')]\n"
        "print(f'  app.py parses OK, executable lines: {len(lines)}')"
    ),
]

for name, code in tests:
    try:
        exec(code)
        passes.append(name)
        print(f'[PASS] {name}')
    except ImportError as e:
        errors.append((name, f'ImportError: {e}'))
        print(f'[SKIP] {name} — missing dep: {e}')
    except AssertionError as e:
        errors.append((name, f'AssertionError: {e}'))
        print(f'[FAIL] {name} — assertion: {e}')
    except Exception as e:
        errors.append((name, str(e)))
        print(f'[FAIL] {name} — {e}')

print()
print(f'Results: {len(passes)}/{len(tests)} passed, {len(errors)} failed/skipped')
if errors:
    print('Failures:')
    for n, e in errors:
        print(f'  [{n}]: {e}')
