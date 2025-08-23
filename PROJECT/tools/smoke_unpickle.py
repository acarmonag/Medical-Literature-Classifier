# tools/smoke_unpickle.py
# Purpose: Verify that joblib artifacts can be loaded in a plain Python session.
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.text_cleaning import MedTextCleaner  # register class
import joblib

pipe = joblib.load(PROJECT_ROOT / "models" / "tfidf_lr_med_pipeline.joblib")
mlb  = joblib.load(PROJECT_ROOT / "models" / "label_binarizer.joblib")
print("OK:", type(pipe).__name__, len(mlb.classes_), "classes")