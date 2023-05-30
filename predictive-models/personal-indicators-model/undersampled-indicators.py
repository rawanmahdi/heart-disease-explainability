
#%%
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from imblearn.under_sampling import RandomUnderSampler
#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/life-heart.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
neg, pos = np.bincount(dataframe['target'])
#%%
df = dataframe.copy()
y = df.pop('target')
X = df
#%%
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test
#%%
# observe class imbalance
neg, pos = np.bincount(y_train)
print("No.negative samples before undersampling",neg)
print("No.positive samples before undersampling",pos)

# %%
# resample via undersampling majority class - this is favoured over oversampling as the dataset is very large
rus = RandomUnderSampler(random_state=0)
rus.fit(X,y)
# only resample training dataset
X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)

neg0, pos0 = np.bincount(y_train_resampled)
print("No.negative samples after undersampling",neg0)
print("No.positive samples after undersampling",pos0)

# %%
