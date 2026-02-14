import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load raster files
raster_files = [
    'param1.tif',
    'param2.tif',
    'param3.tif',
    'param4.tif',
    'param5.tif',
    'param6.tif',
    'param7.tif',
    'param8.tif',
    'param9.tif',
    'param10.tif'
]

# Load and stack .tif files
stacked_data = []
for file in raster_files:
    with rasterio.open(file) as src:
        stacked_data.append(src.read(1))  # Read the first band
stacked_array = np.stack(stacked_data, axis=-1)

# Assuming ground truth labels are in a separate file called `labels.csv`
labels = pd.read_csv('labels.csv')  # Landslide points marked as 1, non-landslide as 0

# Extract features and labels
features = stacked_array.reshape(-1, len(raster_files))
labels = labels.values.flatten()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Generate classification report and AUC-ROC score
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Print evaluation metrics
print(report)
print(f'AUC-ROC: {auc}')

# Create a susceptibility map
predicted_probabilities = rf_model.predict_proba(features)[:, 1]

# Reshape back to original raster dimensions
susceptibility_map = predicted_probabilities.reshape(stacked_array.shape[:2])

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(susceptibility_map, cmap='hot', interpolation='nearest')
plt.colorbar(label='Susceptibility Level')
plt.title('Landslide Susceptibility Map')

# AUC-ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve')
plt.legend()
plt.show()

# Save both the map and the figure
plt.savefig('susceptibility_map.png')
# ... (code for saving the susceptibility map)