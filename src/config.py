from pathlib import Path

# =========================
# PROJECT PATH
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = Path(BASE_DIR / "data" / "raw")

OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

# =========================
# MONGODB CONFIG
# =========================

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "hotel_review_clustering"

PROVINCES_COLLECTION = "provinces"
HOTELS_COLLECTION = "hotels"
REVIEWS_COLLECTION = "reviews"
HOTEL_FEATURES_COLLECTION = "hotel_features"
CLUSTER_RESULTS_COLLECTION = "cluster_results"