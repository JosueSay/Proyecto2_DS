import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")

DATA_DIR = os.path.join(PROJECT_DIR, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")

REPORTS_CLEAN_DIR = os.path.join(PROJECT_DIR, "reports", "clean")
REPORTS_EDA_DIR   = os.path.join(PROJECT_DIR, "reports", "eda")

IMAGES_EDA_DIR = os.path.join(PROJECT_DIR, "images", "eda")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

for d in [REPORTS_CLEAN_DIR, REPORTS_EDA_DIR, IMAGES_EDA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

DIRS = {
    "project_dir": PROJECT_DIR,
    "clean_dir": CLEAN_DIR,
    "reports_clean_dir": REPORTS_CLEAN_DIR,
    "reports_eda_dir": REPORTS_EDA_DIR,
    "images_dir": IMAGES_EDA_DIR,
    "cache_dir": CACHE_DIR,
}
