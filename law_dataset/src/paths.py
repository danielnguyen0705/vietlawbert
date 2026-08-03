import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
JSON_DIR = os.path.join(DATA_DIR, "json")
MD_DIR = os.path.join(DATA_DIR, "markdown") # Phục hồi MD_DIR cho contextualizer

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

# Files phục vụ tracking state
METADATA_FILE = os.path.join(JSON_DIR, "metadata.jsonl")
FAILED_FILE = os.path.join(JSON_DIR, "failed_links.jsonl")

# Thư mục log
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def get_log_path(script_name: str) -> str:
    return os.path.join(LOGS_DIR, f"log_{script_name}.log")

def ensure_dirs():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)
