import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# Configure logging
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

try:
    # Load model building parameters from params.yaml
    params = yaml.safe_load(open('params.yaml', 'r'))['model_building']
    logger.debug(f"Loaded model_building parameters: {params}")
except Exception as e:
    logger.error(f"Error loading model_building parameters from params.yaml: {e}")
    raise

try:
    # Fetch the data from data/features/train_bow.csv
    train_data = pd.read_csv('./data/features/train_bow.csv')
    logger.debug("Loaded train data from train_bow.csv")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except pd.errors.ParserError as e:
    logger.error(f"Error parsing CSV file: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error occurred while loading data: {e}")
    raise

try:
    # Split data into features (X_train) and labels (y_train)
    X_train = train_data.iloc[:, 0:-1].values
    y_train = train_data.iloc[:, -1].values

    # Define and train the Gradient Boosting model
    clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    clf.fit(X_train, y_train)
    logger.debug("Trained GradientBoostingClassifier")
except Exception as e:
    logger.error(f"Error training GradientBoostingClassifier: {e}")
    raise

try:
    # Save the trained model to model.pkl
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    logger.debug("Saved model.pkl")
except Exception as e:
    logger.error(f"Error saving model to model.pkl: {e}")
    raise
