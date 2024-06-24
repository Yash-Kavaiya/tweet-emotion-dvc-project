import numpy as np
import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer
import logging

# Configure logging
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

try:
    # Load max_features from params.yaml
    max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']
    logger.debug(f"Loaded max_features: {max_features}")
except Exception as e:
    logger.error(f"Error loading max_features from params.yaml: {e}")
    raise

try:
    # Fetch the data from data/processed
    train_data = pd.read_csv('./data/processed/train_processed.csv')
    test_data = pd.read_csv('./data/processed/test_processed.csv')
    logger.debug("Loaded train and test data.")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except pd.errors.ParserError as e:
    logger.error(f"Error parsing CSV file: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error occurred while loading data: {e}")
    raise

# Fill NaN values with empty string
train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=max_features)

try:
    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(train_data['content'].values)
    logger.debug("Applied CountVectorizer on training data.")
    
    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(test_data['content'].values)
    logger.debug("Transformed test data using CountVectorizer.")
except Exception as e:
    logger.error(f"Error applying CountVectorizer: {e}")
    raise

try:
    # Create DataFrames with BoW features and labels
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = train_data['sentiment']

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = test_data['sentiment']
    logger.debug("Created train and test DataFrames with BoW features.")
except Exception as e:
    logger.error(f"Error creating DataFrames: {e}")
    raise

# Store the data inside data/features
data_path = os.path.join("data", "features")

try:
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
    logger.debug("Saved train_bow.csv and test_bow.csv in data/features.")
except Exception as e:
    logger.error(f"Error saving CSV files: {e}")
    raise
