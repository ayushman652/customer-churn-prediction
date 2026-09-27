from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DATASET_PATH = WORKSPACE_ROOT / "datasets" / "churn" / "churnData.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET = "churn"

FEATURES = [
    "tenure",
    "age",
    "address",
    "income",
    "ed",
    "employ",
    "equip"
]