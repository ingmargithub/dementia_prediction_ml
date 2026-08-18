import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[0]  # Adjust the parent index if needed
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from .config import INPUT_FILE, RESULTS_DIR, SHEET_NAME

from .evaluation import evaluate_models
from .features import build_feature_sets, prepare_dataframe
from .inference import compare_final_models
from .reporting import save_results
from .missingness import check_missingness, diagnose_features