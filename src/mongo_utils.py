from pymongo import MongoClient
from config import MONGO_URI, DATABASE_NAME


def get_mongo_client():
    client = MongoClient(MONGO_URI)
    return client


def get_database():
    client = get_mongo_client()
    db = client[DATABASE_NAME]
    return db


def get_collection(collection_name):
    db = get_database()
    return db[collection_name]