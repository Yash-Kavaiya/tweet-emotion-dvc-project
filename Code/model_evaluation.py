import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

try:
    # Load the trained classifier
    clf = pickle.load(open('model.pkl', 'rb'))
    logger.debug("Loaded model.pkl")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except pickle.UnpicklingError as e:
    logger.error(f"Error loading model.pkl: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading model.pkl: {e}")
    raise

try:
    # Load test data
    test_data = pd.read_csv('./data/features/test_bow.csv')
    logger.debug("Loaded test data from test_bow.csv")
except FileNotFoundError as e:
    logger.error(f"Test data file not found: {e}")
    raise
except pd.errors.ParserError as e:
    logger.error(f"Error parsing CSV file: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading test data: {e}")
    raise

try:
    # Prepare test data
    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values
except KeyError as e:
    logger.error(f"Missing column in test data: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error preparing test data: {e}")
    raise

try:
    # Predict labels
    y_pred = clf.predict(X_test)

    # Check if classifier has predict_proba method
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)
    else:
        y_pred_proba = None

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Calculate AUC only if y_pred_proba is available
    if y_pred_proba is not None:
        if y_pred_proba.shape[1] == len(np.unique(y_test)):  # Ensure correct shape for multiclass
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            auc = None
    else:
        auc = None

    # Prepare metrics dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    # Save metrics to JSON file
    with open('metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)
    logger.debug("Saved metrics to metrics.json")
except Exception as e:
    logger.error(f"Error during evaluation or saving metrics: {e}")
    raise
