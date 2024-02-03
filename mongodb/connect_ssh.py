import sshtunnel
import pymongo

MONGO_URI = 'mongodb://127.0.0.1:37017'

SSH_SERVER = '168.131.153.58'
SSH_PORT = 2222
SSH_USERNAME = 'andrew'
SSH_PASSWORD = 'a'

with sshtunnel.open_tunnel(
    (SSH_SERVER, SSH_PORT),
    ssh_username=SSH_USERNAME,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=('127.0.0.1', 27017),
    local_bind_address=('0.0.0.0', 37017)
  ) as tunnel:

    # print(tunnel.local_bind_port)

    # connect to mongo uri
    client = pymongo.MongoClient(MONGO_URI)

    # list database names
    names = client.list_database_names()
    print(names)

    farms = client.SmartFarm.farms
    print(farms.find_one({"owner": "Vu"}))