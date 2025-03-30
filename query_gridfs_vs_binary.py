from pymongo import MongoClient
from gridfs import GridFS
from bson import Binary
from PIL import Image
from io import BytesIO
import time

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://anPham:2252011@dbms-assignment.bxeclce.mongodb.net/?retryWrites=true&w=majority&appName=DBMS-Assignment")
db = client["mri_database"]

# Collections
binary_collection = db["mri_images_binary"]
metadata_collection = db["mri_images_gridfs"]

# Initialize GridFS (for gridfs files stored in "mri_images_files")
fs = GridFS(db, collection="mri_images_files")

# Retrieve by patient_id
patient_id = "PA_0003"

# -----------------------------
# Retrieval from the binary collection
start = time.time()
binary_doc = binary_collection.find_one({"patient_id": patient_id})
if binary_doc:
    image_data = binary_doc["image"]
    image = Image.open(BytesIO(image_data))
    # image.show(title=f"Binary Image - {patient_id}")
end = time.time()
print(f"Binary retrieval took: {end - start:.6f} seconds.")
# else:
#     print("No document found in binary collection for", patient_id)

# -----------------------------
# Retrieval from the GridFS collection
start = time.time()
metadata_doc = metadata_collection.find_one({"patient_id": patient_id})
if metadata_doc:
    fs_id = metadata_doc["file_id"]
    gridfs_file = fs.get(fs_id)
    image_data = gridfs_file.read()
    image = Image.open(BytesIO(image_data))
    # image.show(title=f"GridFS Image - {patient_id}")
end = time.time()
print(f"GridFS retrieval took: {end - start:.6f} seconds.")
# else:
#     print("No document found in GridFS metadata collection for", patient_id)