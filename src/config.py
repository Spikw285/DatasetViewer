from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "petrobras 3W main dataset"

EVENTS = {
    0: "Normal",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSB",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}

SENSORS = ["T-TPT", "P-TPT", "P-PDG", "P-MON-CKP", "T-JUS-CKP"]

WINDOW_SIZE = 60
WINDOW_STEP = 30
RANDOM_STATE = 42