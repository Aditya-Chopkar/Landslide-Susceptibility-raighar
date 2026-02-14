import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt

def load_tif_files(tif_folder):
    tif_data = []
    for filename in os.listdir(tif_folder):
        if filename.endswith('.tif'):
            with rasterio.open(os.path.join(tif_folder, filename)) as src:
                tif_data.append(src.read(1))  # Read the first band
    return np.array(tif_data)

def extract_landslide_points(landslides_file):
    # Load landslide points (Assuming they are in a CSV format with lat/lon or similar)
    landslide_points = pd.read_csv(landslides_file)
    return landslide_points

def train_rf_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    return clf, y_test, y_pred

def generate_susceptibility_map(model, raster_data):
    # Implement a logic to generate a 5-class susceptibility map
    # This is a placeholder for a real implementation
    susceptibility_map = model.predict(raster_data.reshape(-1, raster_data.shape[-1]))
    return susceptibility_map.reshape(raster_data.shape[1], raster_data.shape[2])

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def main(tif_folder, landslides_file):
    raster_data = load_tif_files(tif_folder)
    landslide_points = extract_landslide_points(landslides_file)
    
    # Assuming we are using features derived from raster_data
    # For this example, let's assume the labels are in landslide_points
    labels = landslide_points['label']  # Replace with your actual label extraction logic
    features = raster_data.reshape(-1, raster_data.shape[-1])  # Reshape if necessary
    
    model, y_test, y_pred = train_rf_classifier(features, labels)
    susceptibility_map = generate_susceptibility_map(model, raster_data)
    
    plot_roc_curve(y_test, y_pred)
    
if __name__ == '__main__':
    # Define paths
    tif_folder = 'path/to/tif/files'
    landslides_file = 'path/to/landslides.csv'
    
    main(tif_folder, landslides_file)
