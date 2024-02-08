import motor.motor_asyncio
import yaml

with open("config.yaml") as f:
    key_param = yaml.safe_load(f)

    client = motor.motor_asyncio.AsyncIOMotorClient(key_param['MONGO_URI'])
    db = client.college
    student_collection = db.get_collection("students")
