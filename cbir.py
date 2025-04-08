import os
import numpy as np
import cv2
from pymongo import MongoClient
from bson import Binary
from PIL import Image
from io import BytesIO
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://anPham:2252011@dbms-assignment.bxeclce.mongodb.net/?retryWrites=true&w=majority&appName=DBMS-Assignment")
db = client["mri_database"]
collection = db["mri_images"]

# Create a new collection for storing features
features_collection = db["mri_features"]

def extract_features(image_data):
    """
    Extract features from image data using multiple methods:
    1. Histogram of Oriented Gradients (HOG) - for shape features
    2. Local Binary Patterns (LBP) - for texture features
    3. Statistical features - mean, std dev, etc.
    
    Returns a feature vector that represents the image
    """
    # Convert binary data to numpy array for OpenCV processing
    if isinstance(image_data, Binary):
        image_data = image_data

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Resize for consistency (MRI scans might have different dimensions)
    img = cv2.resize(img, (256, 256))

    # 1. HOG Features (shape-based)
    # Using a smaller window size for HOG to be compatible with most OpenCV versions
    win_size = (256, 256)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(img)

    # 2. LBP Features (texture-based) using scikit-image instead of OpenCV
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize

    # 3. Histogram of pixel intensities (simplified texture representation)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 4. Statistical features
    mean = np.mean(img)
    std = np.std(img)
    
    # 5. GLCM texture features using scikit-image
    glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Combine all features
    feature_vector = np.concatenate([
        hog_features.flatten()[:100],  # Take first 100 HOG features
        lbp_hist,                      # LBP histogram features
        hist[:50],                     # Take first 50 histogram bins
        np.array([mean, std]),         # Statistical features
        contrast, dissimilarity, homogeneity, energy, correlation  # GLCM features
    ])

    # Normalize the entire feature vector to unit length
    feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-7)

    return feature_vector

def index_all_images():
    """
    Process all images in the database, extract features, and store them
    """
    # Clear existing features collection
    features_collection.delete_many({})

    # Get all documents from the main collection
    cursor = collection.find()
    count = 0
    start_time = time.time()
    
    for doc in cursor:
        if "image" in doc:
            try:
                # Extract features
                feature_vector = extract_features(doc["image"])

                # Store features with reference to original document
                feature_doc = {
                    "patient_id": doc["patient_id"],
                    "filename": doc["filename"],
                    "diagnosis": doc["diagnosis"],
                    "features": Binary(feature_vector.tobytes()),
                    "feature_dim": len(feature_vector)
                }

                features_collection.insert_one(feature_doc)
                count += 1
                print(f"Processed image {count}: {doc['filename']}")
            except Exception as e:
                print(f"Error processing {doc['filename']}: {e}")
    
    end_time = time.time()
    print(f"Indexed features for {count} images in {end_time - start_time:.2f} seconds")

def search_similar_images(query_image_data, top_n=10, filter_diagnosis=None, query_doc=None):
    """
    Search for similar images based on extracted features
    
    Args:
        query_image_data: Binary image data or filename to search for
        top_n: Number of similar images to return
        filter_diagnosis: Optional filter to only return images with a specific diagnosis
        
    Returns:
        List of similar image documents
    """
    # Extract features from query image
    query_features = extract_features(query_image_data)
    
    # Prepare query filter
    query_filter = {}
    if filter_diagnosis:
        query_filter["diagnosis"] = filter_diagnosis
    
    # Get all feature vectors from the database matching the filter
    feature_docs = list(features_collection.find(query_filter))
    similarities = []
    
    # Compare query features with all stored features
    for doc in feature_docs:
        # Convert stored binary features back to numpy array
        stored_features = np.frombuffer(doc["features"], dtype=np.float64)
        
        # Reshape if needed
        if len(stored_features) != doc["feature_dim"]:
            stored_features = stored_features[:doc["feature_dim"]]
        
        # Add a small check to avoid comparing with the exact same image
        is_same_image = False
        if query_doc and "patient_id" in doc and "patient_id" in query_doc and doc["patient_id"] == query_doc["patient_id"]:
            is_same_image = True

        # Only add to similarities if it's NOT the same image
        if not is_same_image:
            # Calculate similarity
            similarity = cosine_similarity([query_features], [stored_features])[0][0]
            similarities.append((doc, similarity))

    # Sort by similarity (higher is better)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N similar images
    similar_images = []
    for doc, similarity in similarities[:top_n]:
        # Get the full document from the main collection
        full_doc = collection.find_one({"patient_id": doc["patient_id"]})
        if full_doc:
            similar_images.append({
                "similarity": float(similarity),  # Convert from numpy float to Python float
                "patient_id": doc["patient_id"],
                "filename": doc["filename"],
                "diagnosis": doc["diagnosis"]
            })
    
    return similar_images

def visualize_similar_images(query_image_data, similar_images, title="Similar MRI Scans"):
    """
    Visualize the query image and similar images
    """
    # Convert query image to numpy array
    nparr = np.frombuffer(query_image_data, np.uint8)
    query_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Calculate grid dimensions - use a 2x5 grid for 10 similar images or adjust dynamically based on the number of images
    num_similar = len(similar_images)
    if num_similar <= 5:
        grid_rows, grid_cols = 1, num_similar
    else:
        grid_rows = (num_similar + 4) // 5
        grid_cols = min(5, num_similar)

    # Create figure with subplots for the grid
    fig = plt.figure(figsize=(3 + grid_cols * 3, 3 + grid_rows * 3))

    # Create a larger subplot for the query image at the top
    ax_query = plt.subplot2grid((grid_rows + 1, grid_cols), (0, 0), colspan=grid_cols)
    ax_query.imshow(query_img, cmap='gray')
    ax_query.set_title("Query Image", fontsize=14, pad=10)
    ax_query.axis('off')

    # Add diagnosis below the query image title
    if similar_images and 'diagnosis' in similar_images[0]:
        query_diagnosis = similar_images[0]['diagnosis']
        ax_query.text(0.5, -0.05, f"Diagnosis: {query_diagnosis}", 
                      transform=ax_query.transAxes, fontsize=12,
                      ha='center', va='top')

    # Display similar images in a grid
    for i, sim_doc in enumerate(similar_images):
        # Get the image data
        img_doc = collection.find_one({"patient_id": sim_doc["patient_id"]})
        if img_doc and "image" in img_doc:
            # Convert to numpy array
            img_data = img_doc["image"]
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Calculate position in grid (starting from second row)
            row = (i // grid_cols) + 1  # +1 because query image is in row 0
            col = i % grid_cols
            
            # Create subplot
            ax = plt.subplot2grid((grid_rows + 1, grid_cols), (row, col))
            
            # Display image
            ax.imshow(img, cmap='gray')
            
            # Display similarity and diagnosis
            similarity_str = f"Similarity: {sim_doc['similarity']:.2f}"
            diagnosis_str = f"{sim_doc['diagnosis']}"
            
            # Use different colors for different diagnoses
            diagnosis_color = 'green'
            if 'Glioma' in diagnosis_str:
                diagnosis_color = 'red'
            elif 'No tumor' in diagnosis_str:
                diagnosis_color = 'blue'
            elif 'Pituitary' in diagnosis_str:
                diagnosis_color = 'purple'
            
            ax.set_title(similarity_str, fontsize=10, pad=5)
            ax.text(0.5, -0.05, diagnosis_str, transform=ax.transAxes, 
                   color=diagnosis_color, fontsize=9,
                   ha='center', va='top', wrap=True)
            
            # Turn off axis
            ax.axis('off')
    
    # Add a main title
    plt.suptitle(title, fontsize=16, y=0.98)
    
    # Add a legend for diagnoses
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Meningioma'),
        Line2D([0], [0], color='red', lw=4, label='Glioma'),
        Line2D([0], [0], color='blue', lw=4, label='No tumor'),
        Line2D([0], [0], color='purple', lw=4, label='Pituitary')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=4, frameon=False, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and show
    plt.savefig("similar_images.png", dpi=300, bbox_inches='tight')
    plt.show()

def compare_diagnoses():
    """
    Compare image features across different diagnoses
    """
    # Get all distinct diagnoses
    diagnoses = collection.distinct("diagnosis")
    
    # Set up the plot
    fig, axes = plt.subplots(len(diagnoses), 3, figsize=(15, 5 * len(diagnoses)))
    
    for i, diagnosis in enumerate(diagnoses):
        # Get a sample image for this diagnosis
        sample_doc = collection.find_one({"diagnosis": diagnosis})
        if sample_doc and "image" in sample_doc:
            # Get the image data
            img_data = sample_doc["image"]
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Display the original image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f"{diagnosis} - Original")
            axes[i, 0].axis('off')
            
            # Display edge detection (shape feature visualization)
            edges = cv2.Canny(img, 100, 200)
            axes[i, 1].imshow(edges, cmap='gray')
            axes[i, 1].set_title(f"{diagnosis} - Edge Detection")
            axes[i, 1].axis('off')
            
            # Display LBP (texture feature visualization)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            axes[i, 2].imshow(lbp, cmap='gray')
            axes[i, 2].set_title(f"{diagnosis} - LBP Texture")
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("diagnosis_comparison.png")
    plt.show()

def evaluate_medical_relevance(query_doc, similar_images):
    """
    Calculate medical relevance metrics for CBIR results
    
    Args:
        query_doc: The document containing the query image
        similar_images: List of similar images returned by CBIR
        
    Returns:
        Dictionary of metrics
    """
    query_diagnosis = query_doc["diagnosis"]
    
    # Count images with matching diagnosis
    matching_diagnosis = sum(1 for img in similar_images if img["diagnosis"] == query_diagnosis)
    
    # Calculate diagnostic precision (percentage of results with same diagnosis)
    diagnostic_precision = (matching_diagnosis / len(similar_images)) * 100 if similar_images else 0
    
    # Calculate precision at different ranks (P@k)
    precision_at_1 = 100 if similar_images and similar_images[0]["diagnosis"] == query_diagnosis else 0
    precision_at_3 = sum(1 for img in similar_images[:3] if img["diagnosis"] == query_diagnosis) / 3 * 100 if len(similar_images) >= 3 else 0
    precision_at_5 = sum(1 for img in similar_images[:5] if img["diagnosis"] == query_diagnosis) / 5 * 100 if len(similar_images) >= 5 else 0
    
    # Calculate Mean Reciprocal Rank (MRR)
    # This measures where the first relevant result appears
    first_relevant = next((i+1 for i, img in enumerate(similar_images) if img["diagnosis"] == query_diagnosis), 0)
    mrr = 1/first_relevant if first_relevant > 0 else 0
    
    # Calculate Mean Average Precision (MAP)
    # This considers both precision and ranking of relevant results
    relevant_positions = [i+1 for i, img in enumerate(similar_images) if img["diagnosis"] == query_diagnosis]
    if relevant_positions:
        avg_precision = sum((i+1)/(pos) for i, pos in enumerate(relevant_positions)) / len(relevant_positions)
    else:
        avg_precision = 0
    
    # Calculate Mean Diagnosis Similarity
    # Even between different diagnoses, some may be more related than others
    # E.g., different tumor types may be more related than tumor vs. no tumor
    diagnosis_similarity_map = {
        "Glioma detected": {"Glioma detected": 1.0, "Meningioma detected": 0.5, "Pituitary detected": 0.5, "No tumor": 0.1},
        "Meningioma detected": {"Glioma detected": 0.5, "Meningioma detected": 1.0, "Pituitary detected": 0.5, "No tumor": 0.1},
        "Pituitary detected": {"Glioma detected": 0.5, "Meningioma detected": 0.5, "Pituitary detected": 1.0, "No tumor": 0.1},
        "No tumor": {"Glioma detected": 0.1, "Meningioma detected": 0.1, "Pituitary detected": 0.1, "No tumor": 1.0}
    }
    
    mean_diagnosis_similarity = 0
    for img in similar_images:
        if query_diagnosis in diagnosis_similarity_map and img["diagnosis"] in diagnosis_similarity_map[query_diagnosis]:
            mean_diagnosis_similarity += diagnosis_similarity_map[query_diagnosis][img["diagnosis"]]
    
    mean_diagnosis_similarity = mean_diagnosis_similarity / len(similar_images) * 100 if similar_images else 0
    
    # Return all metrics
    return {
        "diagnostic_precision": diagnostic_precision,
        "precision_at_1": precision_at_1,
        "precision_at_3": precision_at_3,
        "precision_at_5": precision_at_5,
        "mean_reciprocal_rank": mrr,
        "mean_average_precision": avg_precision,
        "mean_diagnosis_similarity": mean_diagnosis_similarity,
        "matching_diagnosis_count": matching_diagnosis,
        "total_results": len(similar_images)
    }

def visualize_medical_metrics(metrics, title="CBIR Medical Relevance Metrics"):
    """
    Create a visual representation of the medical relevance metrics
    """
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Precision metrics in first subplot (bar chart)
    precision_metrics = [
        metrics["diagnostic_precision"], 
        metrics["precision_at_1"], 
        metrics["precision_at_3"], 
        metrics["precision_at_5"],
        metrics["mean_diagnosis_similarity"]
    ]
    precision_labels = [
        "Overall\nPrecision", 
        "Precision@1", 
        "Precision@3", 
        "Precision@5",
        "Diagnosis\nSimilarity"
    ]
    bars = ax1.bar(precision_labels, precision_metrics, color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6'])
    
    # Add labels and title
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Precision Metrics')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Ranking metrics in second subplot (gauge-like visualization)
    ranking_metrics = [
        metrics["mean_reciprocal_rank"],
        metrics["mean_average_precision"]
    ]
    ranking_labels = [
        "Mean Reciprocal\nRank (MRR)",
        "Mean Average\nPrecision (MAP)"
    ]
    
    # Create horizontal bars that look like gauges
    max_val = 1.0
    bars = ax2.barh(ranking_labels, ranking_metrics, color=['#3498db', '#2ecc71'], height=0.4)
    ax2.barh(ranking_labels, [max_val] * len(ranking_metrics), color='#ecf0f1', height=0.4, alpha=0.3)
    
    # Add labels and title
    ax2.set_xlim(0, max_val + 0.1)
    ax2.set_title('Ranking Metrics')
    ax2.set_xlabel('Score (higher is better)')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', va='center')
    
    # Add a subtitle explaining the metrics
    plt.figtext(0.5, 0.01, 
                "These metrics evaluate clinical relevance of CBIR results.\n"
                "Higher values indicate better alignment between visual similarity and diagnostic relevance.",
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    # Add the main title
    plt.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save and show
    plt.savefig("medical_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

def demo_cbir():
    """
    Demonstrate the CBIR functionality with a sample query
    """
    # First, make sure all images are indexed
    print("Indexing all images (this may take a while)...")
    index_all_images()
    
    # Select a sample image as query
    sample_doc = collection.find_one({"diagnosis": "Meningioma detected"})
    if not sample_doc:
        print("No sample image found")
        return
    
    query_image = sample_doc["image"]
    
    print("\n=== Basic CBIR Search ===")
    # Search for similar images
    similar_images = search_similar_images(query_image, top_n=10, query_doc=sample_doc)
    
    # Display results
    print("\nQuery Image:")
    print(f"Patient ID: {sample_doc['patient_id']}")
    print(f"Filename: {sample_doc['filename']}")
    print(f"Diagnosis: {sample_doc['diagnosis']}")
    
    print("\nSimilar Images:")
    for i, img in enumerate(similar_images):
        print(f"\n{i+1}. Similarity: {img['similarity']:.4f}")
        print(f"   Patient ID: {img['patient_id']}")
        print(f"   Filename: {img['filename']}")
        print(f"   Diagnosis: {img['diagnosis']}")

    # Calculate and display medical relevance metrics
    print("\n=== Medical Relevance Metrics ===")
    metrics = evaluate_medical_relevance(sample_doc, similar_images)
    print(f"Diagnostic Precision: {metrics['diagnostic_precision']:.2f}% ({metrics['matching_diagnosis_count']} of {metrics['total_results']} images match the query diagnosis)")
    print(f"Precision@1: {metrics['precision_at_1']:.2f}% (is the top result clinically relevant?)")
    print(f"Precision@3: {metrics['precision_at_3']:.2f}% (percentage of top 3 results that are clinically relevant)")
    print(f"Precision@5: {metrics['precision_at_5']:.2f}% (percentage of top 5 results that are clinically relevant)")
    print(f"Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f} (higher is better, max 1.0)")
    print(f"Mean Average Precision: {metrics['mean_average_precision']:.4f} (higher is better, max 1.0)")
    print(f"Mean Diagnosis Similarity: {metrics['mean_diagnosis_similarity']:.2f}% (accounts for related diagnoses)")

    # After calculating metrics
    visualize_medical_metrics(metrics)
    
    # Visualize the results
    visualize_similar_images(query_image, similar_images, "Similar MRI Scans (All Diagnoses)")
    
    print("\n=== Filtered CBIR Search ===")
    # Search for similar images with the same diagnosis
    similar_same_diagnosis = search_similar_images(
        query_image, 
        top_n=10, 
        filter_diagnosis=sample_doc["diagnosis"],
        query_doc=sample_doc
    )
    
    # Visualize results filtered by diagnosis
    visualize_similar_images(
        query_image, 
        similar_same_diagnosis, 
        f"Similar MRI Scans (Filtered by {sample_doc['diagnosis']})"
    )
    
    print("\n=== Diagnosis Comparison ===")
    # Compare visual features across different diagnoses
    compare_diagnoses()

if __name__ == "__main__":
    demo_cbir()