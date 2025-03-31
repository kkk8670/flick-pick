import sys
from pathlib import Path


ROOT_DIR = Path(__file__).parent.resolve()

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))