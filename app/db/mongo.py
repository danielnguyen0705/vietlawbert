from pymongo import MongoClient
from app.config import settings

def get_mongo_client():
    uri = (
        f"mongodb://{settings.MONGO_INITDB_ROOT_USERNAME}:"
        f"{settings.MONGO_INITDB_ROOT_PASSWORD}@"
        f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
        f"?authSource={settings.MONGO_AUTH_SOURCE}"
    )
    return MongoClient(uri)

def get_database():
    client = get_mongo_client()
    return client[settings.MONGO_DB_NAME]