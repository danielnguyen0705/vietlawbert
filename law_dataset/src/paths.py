import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == 'posix':
    LOGS_DIR = os.path.expanduser("~/.cache/vietlawbert/logs")
else:
    LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def get_log_path(script_name: str) -> str:
    return os.path.join(LOGS_DIR, f"log_{script_name}.log")

def ensure_dirs():
    os.makedirs(LOGS_DIR, exist_ok=True)
