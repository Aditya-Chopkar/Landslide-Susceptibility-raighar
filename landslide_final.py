import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def load_data(filepath):
    """Load and return the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data."""
    # Handle missing values and categorical variables
    df.fillna(df.mean(), inplace=True)
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])  # Assuming 'class' is the target variable
    return df

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # AUC-ROC
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    print("AUC: ", auc)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plotting the AUC-ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def classify_and_map(model, X, le):
    """Classify the data and map results."""
    predictions = model.predict(X)
    predicted_classes = le.inverse_transform(predictions)
    
    # Assuming there's a function to map predictions back to GIS data (not implemented here)
    return predicted_classes

# Main execution workflow
if __name__ == "__main__":
    # Load dataset
    data = load_data('path_to_your_data.csv')  # Specify the correct path to your data
    data = preprocess_data(data)

    # Split data
    X = data.drop('class', axis=1)  # Features
    y = data['class']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Classify
    classification_results = classify_and_map(model, X, LabelEncoder())