import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import zipfile

# Load TIF Parameter Files
parameters = ['Aspect', 'Curvature', 'DEM', 'Distance_from_Road', 'Geomorphology', 'LULC', 'Lineament', 'Lithology', 'Rainfall', 'Slope']

# Function to read TIF files
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

# Load raster data into a dictionary
raster_data = {param: read_tif(f'{param}.tif') for param in parameters}

# Extract landslide and non-landslide points from ZIP files
landslide_points = []
non_landslide_points = []

# Unzipping and loading data
with zipfile.ZipFile('landslide_data.zip', 'r') as zip_ref:
    zip_ref.extractall('landslide_data')
    for file in os.listdir('landslide_data'):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join('landslide_data', file))
            if 'landslide' in file:
                landslide_points.append(data)
            else:
                non_landslide_points.append(data)

# Prepare Data
X = []
Y = []

for i in range(len(landslide_points)):
    for point in landslide_points[i].itertuples():
        X.append([raster_data[param][point.x, point.y] for param in parameters])
        Y.append(1)  # Landslide
for i in range(len(non_landslide_points)):
    for point in non_landslide_points[i].itertuples():
        X.append([raster_data[param][point.x, point.y] for param in parameters])
        Y.append(0)  # Non-Landslide

X = np.array(X)
Y = np.array(Y)

# Train Random Forest Classifier
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# Generate Susceptibility Map
susceptibility_pred = model.predict(X)

# Map predictions to classes
class_map = np.digitize(susceptibility_pred, bins=[0.2, 0.4, 0.6, 0.8]) # Mapping to 5 classes

# Create AUC-ROC Curve
fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('outputs/roc_curve.png')

# Save Susceptibility Map
output_map = np.zeros_like(class_map)
for i, pred in enumerate(class_map):
    output_map[i] = pred

with rasterio.open('outputs/susceptibility_map.tif', 'w', driver='GTiff', height=output_map.shape[0], width=output_map.shape[1], count=1, dtype=output_map.dtype) as dst:
    dst.write(output_map, 1)