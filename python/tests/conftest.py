from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))