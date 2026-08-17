# EMPTY EXAMPLE CONFIG: OWN INPUT_PATH HIDDEN IN GITIGNORE

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = Path(r"path\to\data")
SHEET_NAME = 0
N_OUTER = 5
RANDOM_STATE = 2026
RESULTS_DIR = PROJECT_ROOT / "results"
