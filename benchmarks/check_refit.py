"""Use the frozen historical source to reproduce Table 1."""
import runpy
from pathlib import Path
import sys
folder = Path(__file__).resolve().parent / 'legacy'
sys.path.insert(0, str(folder))
runpy.run_path(str(folder / 'check_refit.py'), run_name='__main__')
