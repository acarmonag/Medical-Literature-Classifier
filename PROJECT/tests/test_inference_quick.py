from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_groups

def test_single_inference_runs():
    res = predict_groups("Effects of IL-6 in COVID-19",
                         "Randomized trial shows elevated alpha-synuclein; see Fig. 2 [1].")
    assert "labels" in res and isinstance(res["labels"], list)
