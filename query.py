from pymongo import MongoClient

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://anPham:2252011@dbms-assignment.bxeclce.mongodb.net/?retryWrites=true&w=majority&appName=DBMS-Assignment")
db = client["mri_database"]
collection = db["mri_images"]

doc = collection.find_one({"filename": "Tr-pi_0011.jpg"})

if doc and "image" in doc:
    with open("retrieved_image.jpg", "wb") as out_file:
        out_file.write(doc["image"])
        print("Image retrieved and saved.")
else:
    print("Image not found in the database.")