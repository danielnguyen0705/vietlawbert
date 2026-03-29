import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
    MONGO_INITDB_ROOT_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME", "admin")
    MONGO_INITDB_ROOT_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "admin123")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "vietlawbert")
    MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE", "admin")

settings = Settings()