from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_SOURCES = [BASE_DIR / "Housing.csv"]

MODEL_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "predictions"
MODEL_PATH = MODEL_DIR / "saved_model.pkl"
PREDICTIONS_PATH = PREDICTIONS_DIR / "new_houses.csv"

TARGET_COLUMN = "price"
REFERENCE_YEAR = 2026
RANDOM_STATE = 42
TEST_SAMPLE_SIZE = 5

NUMERIC_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

GRID_SEARCH_PARAMS = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 12, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}
