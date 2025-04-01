from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.resolve()
parent_dir = ROOT_DIR.parent

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

__all__ = ['ROOT_DIR']

# print(ROOT_DIR)