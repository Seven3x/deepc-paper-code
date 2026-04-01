from pathlib import Path


DEEP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DEEP_DIR / "Results"
PLOTS_DIR = DEEP_DIR / "Plots"
MOVIES_DIR = DEEP_DIR / "Movies"


def ensure_output_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
