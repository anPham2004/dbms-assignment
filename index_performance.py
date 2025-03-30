import time
from pymongo import MongoClient, ASCENDING
from datetime import datetime

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://anPham:2252011@dbms-assignment.bxeclce.mongodb.net/?retryWrites=true&w=majority&appName=DBMS-Assignment")
db = client["mri_database"]
collection = db["mri_images"]

def time_query():
    start = time.time()
    # Sample query: find documents for a given scan_date range
    query = {
        "scan_date": {
            "$gte": datetime(2020, 1, 1),
            "$lt": datetime(2022, 1, 1)
        }
    }
    docs = list(collection.find(query))
    end = time.time()
    print(f"Query returned {len(docs)} documents in {end - start:.6f} seconds.")

print("Performance BEFORE indexing:")
time_query()

# Create indexes on frequently queried fields
collection.create_index([("scan_date", ASCENDING)])
collection.create_index("patient_id")
collection.create_index("diagnosis")
print("\nIndexes created on 'scan_date', 'patient_id', and 'diagnosis'.")

print("\nPerformance AFTER indexing:")
time_query()