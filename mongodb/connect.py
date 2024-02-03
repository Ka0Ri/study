from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# Replace the placeholder with your Atlas connection string
uri = "mongodb+srv://dtvu1707:0@cluster0.mdv28fx.mongodb.net/?retryWrites=true&w=majority"
# Set the Stable API version when creating a new client
client = MongoClient(uri, server_api=ServerApi('1'))
                          
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.sample_mflix
movies = db.movies
# Find samples
number_of_movies_with_korean = movies.count_documents({"languages": "Korean"})
number_of_movies_with_english = movies.count_documents({"languages": "English"})
print(number_of_movies_with_english,
        number_of_movies_with_korean)

