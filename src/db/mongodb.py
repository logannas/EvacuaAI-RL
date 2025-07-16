import pymongo
from functools import cache
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COL = os.getenv("MONGODB_COL")

@cache
def get_mongodb():
    client = pymongo.MongoClient(f"mongodb://{MONGODB_URI}/")
    db = client[MONGODB_DB]
    collection = db[MONGODB_COL]
    return client, db, collection
