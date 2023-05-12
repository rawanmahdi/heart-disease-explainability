import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
#import seaborn as sns
#from pylab import rcParams
#import matplotlib.pyplot as plt
#from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/base-model/data/heart.csv'
dataset = pd.read_csv(heart_csv_path)
print(dataset.describe())
print(dataset.shape)
dataset.drop_duplicates()
print(dataset.shape)


feature_columns = []

# # numeric columns - real values 
# for column in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
#     feature_columns.append(tf.feature_column.numeric_column(column))

# # bucketized 
# age = tf.feature_column.numeric_column('age')
# age_buckets = tf.feature_column.bucketized_column(age,boundaries=[29, 35, 40, 45, 50, 55, 60, 65])
# feature_columns.append(age_buckets)

# # indicator columns
# dataset["thal"] = dataset['thal'].apply(str) 
# thal = tf.feature_column.categorical_column_with_vocabulary_list(
#     'thal', ['3', '6', '7'])
# one_hot_thal = tf.feature_column.indicator_column(thal)
# feature_columns.append(one_hot_thal)

# dataset["sex"] = dataset['sex'].apply(str) 
# sex = tf.feature_column.categorical_column_with_vocabulary_list(
#     'sex', ['0', '1'])
# one_hot_sex = tf.feature_column.indicator_column(sex)
# feature_columns.append(one_hot_sex)

# dataset["cp"] = dataset['cp'].apply(str) 
# cp = tf.feature_column.categorical_column_with_vocabulary_list(
#     'cp', ['0', '1', '2', '3'])
# one_hot_cp = tf.feature_column.indicator_column(cp)
# feature_columns.append(one_hot_cp)

# dataset["slope"] = dataset['slope'].apply(str) 
# slope = tf.feature_column.categorical_column_with_vocabulary_list(
#     'slope', ['0', '1', '2'])
# one_hot_slope = tf.feature_column.indicator_column(slope)
# feature_columns.append(one_hot_slope)

# # embedding column
# thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
# feature_columns.append(thal_embedding)

# # crossed column
# age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
# age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
# feature_columns.append(age_thal_crossed)

# cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)
# cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
# feature_columns.append(cp_slope_crossed)


# # function to build data pipeline to extract, shuffle and batch load the data
# def create_dataset(df, batch_size=32):
#     df = df.copy()
#     labels = df.pop('target')
#     tf_dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
#     shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
#     return shuffled_tf_dataset.batch(batch_size) # returning 32 samples per batch

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)

# train, test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)

# train_dataset = create_dataset(train)
# test_dataset = create_dataset(test)

# to view 3 batches of data: 
# for person in train_dataset.take(3):
#     print(person[0]) #  the input dictionaries
#     print(person[1]) #  the binary categorical labels

# for unit in train.get('target'):
#     print(unit)


# model = tf.keras.models.Sequential([
#     tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
#     tf.keras.layers.Dense(units=128, activation='relu'), 
#     tf.keras.layers.Dropout(rate=0.2),
#     tf.keras.layers.Dense(units=128, activation='relu'), 
#     # tf.keras.layers.Dense(units=128, activation='relu'), 
#     # tf.keras.layers.Dropout(rate=0.2),
#     # tf.keras.layers.Dense(units=60, activation='relu'),
#     tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics = ['accuracy'])

# history = model.fit(train_dataset, 
#                     validation_data=test_dataset, 
#                     epochs=100,
#                     use_multiprocessing=True, verbose=1)

# predictions = model.predict(test_dataset)
# binary_predictions = tf.round(predictions).numpy().flatten()


# print(classification_report(test.get('target'), binary_predictions))
