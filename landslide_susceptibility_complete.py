import numpy as np
import pandas as pd
import rasterio
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def load_tif_files(tif_folder):
    """ Load all TIF files from a folder. """
    rasters = []
    for file in os.listdir(tif_folder):
        if file.endswith('.tif'):
            with rasterio.open(os.path.join(tif_folder, file)) as src:
                rasters.append(src.read(1))
    return np.array(rasters)

def extract_landslide_points(data):
    """Extracts landslide points from the dataset."""
    # Dummy implementation, replace with actual extraction logic
    return np.random.rand(100, 2), np.random.randint(0, 2, 100)

def train_model(X, y):
    """ Trains a Random Forest Classifier. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def plot_auc(y_test, y_scores):
    """ Creates and saves AUC-ROC curve. """
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('roc_curve.png')

def main():
    try:
        tif_folder = 'path/to/tif/files'
        data = load_tif_files(tif_folder)
        landslide_points, labels = extract_landslide_points(data)
        
        model, X_test, y_test = train_model(landslide_points, labels)
        
        y_scores = model.predict_proba(X_test)[:, 1]
        plot_auc(y_test, y_scores)

        print("Processing complete. Outputs saved.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()