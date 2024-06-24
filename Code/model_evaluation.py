import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the classifier
clf = pickle.load(open('model.pkl', 'rb'))

# Load the test data
test_data = pd.read_csv('./data/features/test_bow.csv')
X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# Predict the labels and probabilities
y_pred = clf.predict(X_test)

# Check if classifier has predict_proba method
if hasattr(clf, "predict_proba"):
    y_pred_proba = clf.predict_proba(X_test)
else:
    y_pred_proba = None

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Changed
recall = recall_score(y_test, y_pred, average='weighted')        # Changed

# Calculate AUC only if y_pred_proba is available
if y_pred_proba is not None:
    if y_pred_proba.shape[1] == len(np.unique(y_test)):  # Ensure correct shape for multiclass
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:
        auc = None
else:
    auc = None

metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}

# Save metrics to JSON file
with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)
