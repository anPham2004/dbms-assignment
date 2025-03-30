import os
import time
from pymongo import MongoClient, ASCENDING
from bson import Binary
from gridfs import GridFS
from datetime import datetime, timedelta
import random

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://anPham:2252011@dbms-assignment.bxeclce.mongodb.net/?retryWrites=true&w=majority&appName=DBMS-Assignment")
db = client["mri_database"]

# 02 collections:
binary_collection = db["mri_images_binary"]
metadata_collection = db["mri_images_gridfs"]

# Initialize GridFS
fs = GridFS(db, collection="mri_images_files")

# Define folder path and list of valid files
folder_path = "/Users/anpham/Downloads/HK242/DBMS/BTL/dataset"
valid_files = [f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith((".jpg", ".jpeg"))]

# -----------------------------
# Some helper functions
def detect_diagnosis(filename):
    lower_fname = filename.lower()
    if "tr-gl" in lower_fname:
        return "Glioma detected"
    elif "tr-me" in lower_fname:
        return "Meningioma detected"
    elif "tr-pi" in lower_fname:
        return "Pituitary detected"
    else:
        return "No tumor"

def random_date(start_date, end_date):
    time_diff = end_date - start_date
    random_days = random.randint(0, time_diff.days)
    return start_date + timedelta(days=random_days)

def random_age():
    return random.randint(18, 80)

def random_gender():
    return random.choice(["Male", "Female"])

start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 1, 1)

# # Automatically generate a unique patient_id by incrementing starting from the current max in the collection.
def get_current_max_patient(coll):
    max_patient = 0
    for doc in coll.find({}, {"patient_id": 1}):
        try:
            n = int(doc["patient_id"].split("_")[1])
            if n > max_patient:
                max_patient = n
        except Exception:
            continue
    return max_patient

# -----------------------------
# Insertion using traditional binary method
def insert_binary():
    current_max = get_current_max_patient(binary_collection)
    start_time = time.time()
    for filename in valid_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as image_file:
            image_binary = Binary(image_file.read())
        
        current_max += 1
        patient_id = f"PA_{current_max:04d}"
        document = {
            "filename": filename,
            "image": image_binary,
            "patient_id": patient_id,
            "scan_date": random_date(start_date, end_date),
            "patient_age": random_age(),
            "patient_gender": random_gender(),
            "diagnosis": detect_diagnosis(filename),
            "modality": "MRI"
        }
        binary_collection.insert_one(document)
    end_time = time.time()
    print(f"Binary insertion of {len(valid_files)} files took {end_time - start_time:.6f} seconds.")

# -----------------------------
# Insertion using GridFS
def insert_gridfs():
    # We'll store metadata separately and image files in GridFS.
    current_max = get_current_max_patient(metadata_collection)
    start_time = time.time()
    for filename in valid_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # Store image in GridFS and get the file id
        fs_id = fs.put(image_bytes, filename=filename)
        
        current_max += 1
        patient_id = f"PA_{current_max:04d}"
        document = {
            "filename": filename,
            "file_id": fs_id, 
            "patient_id": patient_id,
            "scan_date": random_date(start_date, end_date),
            "patient_age": random_age(),
            "patient_gender": random_gender(),
            "diagnosis": detect_diagnosis(filename),
            "modality": "MRI"
        }
        metadata_collection.insert_one(document)
    end_time = time.time()
    print(f"GridFS insertion of {len(valid_files)} files took {end_time - start_time:.6f} seconds.")

if __name__ == "__main__":
    print("Starting binary insertion test...")
    insert_binary()
    print("\nStarting GridFS insertion test...")
    insert_gridfs()