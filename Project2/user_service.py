# user_service.py
from pymongo import MongoClient
import os

# MongoDB 연결 설정
db_client = MongoClient("mongodb+srv://yoonsun2596:qwer1234@tm2.hl7a3.mongodb.net/")
db = db_client['your_database_name']
users_collection = db['users']

def save_user(id, password, nickname, email, birthdate):
    user_data = {
        "id": id,
        "password": password,
        "nickname": nickname,
        "email": email,
        "birthdate": birthdate
    }
    users_collection.insert_one(user_data)

def check_user(id, password):
    user = users_collection.find_one({"id": id, "password": password})
    if user:
        return user['nickname']
    return None
