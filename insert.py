import os
from pymongo import MongoClient
from bson import Binary
from datetime import datetime, timedelta
import random

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://anPham:2252011@dbms-assignment.bxeclce.mongodb.net/?retryWrites=true&w=majority&appName=DBMS-Assignment")
db = client["mri_database"]
collection = db["mri_images"]

# Define folder path
folder_path = "/Users/anpham/Downloads/HK242/DBMS/BTL/dataset"

# -----------------------------
# Determine current max patient_id number in DB
max_patient = 0
for doc in collection.find({}, {"patient_id": 1}):
    try:
        n = int(doc["patient_id"].split("_")[1])
        if n > max_patient:
            max_patient = n
    except Exception:
        continue

# -----------------------------
# Get valid image files in folder
valid_files = [f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith((".jpg", ".jpeg"))]

# -----------------------------
# Detect diagnosis based on filename
def detect_diagnosis(filename):
    # Convert filename to lowercase for case-insensitive comparison
    lower_fname = filename.lower()
    if "tr-gl" in lower_fname:
        return "Glioma detected"
    elif "tr-me" in lower_fname:
        return "Meningioma detected"
    elif "tr-pi" in lower_fname:
        return "Pituitary detected"
    else:
        return "No tumor"

# -----------------------------
# Random metadata
def random_date(start_date, end_date):
    time_diff = end_date - start_date
    random_days = random.randint(0, time_diff.days)
    return start_date + timedelta(days=random_days)

start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 1, 1)

def random_age():
    return random.randint(18, 80)

def random_gender():
    return random.choice(["Male", "Female"])

# -----------------------------
# Loop through valid images then assign automatic patient_id and metadata
for filename in valid_files:
    file_path = os.path.join(folder_path, filename)

    # Read image from file
    with open(file_path, "rb") as image_file:
        image_binary = Binary(image_file.read())

    # Increment patient number automatically
    max_patient += 1
    patient_id = f"PA_{max_patient:04d}"

    # If only one image is present, ask for manual metadata entry
    if len(valid_files) == 1:
        print("Manual metadata entry for single image:")
        scan_date_input = input("Enter scan date (YYYY-MM-DD HH:MM:SS): ")
        try:
            scan_date = datetime.strptime(scan_date_input, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Invalid date format. Using a random date instead.")
            scan_date = random_date(start_date, end_date)
        try:
            patient_age = int(input("Enter patient age: "))
        except ValueError:
            print("Invalid age. Using a random age instead.")
            patient_age = random_age()
        patient_gender = input("Enter patient gender (Male/Female): ")
        if patient_gender not in ["Male", "Female"]:
            print("Invalid gender. Using a random gender instead.")
            patient_gender = random_gender()

        # Diagnosis automatically detected from filename
        diagnosis = detect_diagnosis(filename)
        modality = "MRI"
    else:
        # Use random values for metadata when multiple images are in the folder
        scan_date = random_date(start_date, end_date)
        patient_age = random_age()
        patient_gender = random_gender()
        # You can change the default diagnosis or randomize among available options:
        diagnosis = detect_diagnosis(filename)
        modality = "MRI"

    document = {
        "filename": filename,
        "image": image_binary, 
        "patient_id": patient_id, 
        "scan_date": scan_date,
        "patient_age": patient_age,
        "patient_gender": patient_gender,        
        "diagnosis": diagnosis,
        "modality": modality,
    }

    collection.insert_one(document)
    print(f"Inserted: {filename} with Patient_id: {patient_id} and Scan_date: {scan_date}")