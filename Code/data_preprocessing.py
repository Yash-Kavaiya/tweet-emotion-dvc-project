import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from typing import Any, Union
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

import logging

# Configure logging
logger = logging.getLogger('data_transformation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Fetch the data from data/raw
try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.debug("Data loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except pd.errors.ParserError as e:
    logger.error(f"Error parsing CSV file: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error occurred while loading data: {e}")
    raise

# Transform the data
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        raise

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in removing stop words: {e}")
        raise

def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error(f"Error in removing numbers: {e}")
        raise

def lower_case(text: str) -> str:
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in converting to lower case: {e}")
        raise

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"Error in removing punctuations: {e}")
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error in removing URLs: {e}")
        raise

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logger.error(f"Error in removing small sentences: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        return df
    except Exception as e:
        logger.error(f"Error in normalizing text: {e}")
        raise

try:
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    logger.debug("Text normalization completed.")
except Exception as e:
    logger.error(f"Error in text normalization process: {e}")
    raise

# Store the data inside data/processed
data_path = os.path.join("data", "processed")

try:
    os.makedirs(data_path, exist_ok=True)
    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    logger.debug("Processed data saved successfully.")
except Exception as e:
    logger.error(f"Error in saving processed data: {e}")
    raise
