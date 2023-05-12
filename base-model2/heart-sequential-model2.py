import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/base-model2/data/heart2.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# function to build data pipeline to extract, shuffle and batch load the data
def df_to_dataset(df, batch_size=32):
    df = df.copy()
    labels = df.pop('target')
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size) # returning 32 samples per batch


# to view 3 batches of data: 
# for person in train_dataset.take(3):
#     print(person[0]) #  the input dictionaries
#     print(person[1]) #  the binary categorical labels

# for unit in train.get('target'):
#     print(unit)


def get_normalization_layer(feature_name, dataset):
    # normalize numeric features
    normalizer = layers.Normalization(axis=None)
    # extract feature from dataset
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    normalizer.adapt(feature_ds)
    return normalizer

def get_category_encoding_layer(feature_name, dataset, dtype, max_tokens=None):
    if dtype =='string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)
    # extract feature from dataset
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    # 'learn' all possible feature values, assign each an int index 
    index.adapt(feature_ds)
    # encode integer index
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    # multi-hot encode indeices - lambda function captures layers
    return lambda feature: encoder(index(feature))


batch_size=5

#train, test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, batch_size=batch_size)
test_ds = df_to_dataset(test, batch_size=batch_size)
print(len(train_ds))
print(len(val_ds))
print(len(test_ds))

# test out keras preprocessing layers
# view a batch
[(train_features, label_batch)] = train_ds.take(1)
# normalization
bmi_col = train_features['bmi']
layer = get_normalization_layer('bmi', train_ds)
print(layer(bmi_col))
# catgorization
age_col = train_features['ageGroup']
layer = get_category_encoding_layer('ageGroup', train_ds, 'string', max_tokens=14)
print(layer(age_col))

# preprocess all features:


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
