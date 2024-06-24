import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.drop(columns=['tweet_id'],inplace=True)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=421)
print(df.sentiment.value_counts())
data_path = os.path.join('data', 'raw')
os.makedirs(data_path, exist_ok=True)
train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)