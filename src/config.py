from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "petrobras 3W main dataset"

EVENTS = {
    0: "Normal",
    3: "DHSV Failure",
    4: "Severe Slugging",
    7: "Scaling PCK",
    9: "Hydrate",
}

SENSORS = ["T-TPT", "P-TPT", "P-PDG", "P-MON-CKP", "T-JUS-CKP"]

WINDOW_SIZE = 60
WINDOW_STEP = 30
RANDOM_STATE = 42