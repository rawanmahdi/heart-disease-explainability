
#%%
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/base-model2/data/heart2.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
neg, pos = np.bincount(dataframe['target'])

#%%
# function to build data pipeline to extract, shuffle and batch load the data
# def resampled_df(df):

    # pos_df = df[df['target'] == 1]
    # neg_df = df[df['target'] == 0]
    # # pos_labels = pos_df.pop('target')
    # # pos_features = pos_df
    # # neg_labels = neg_df.pop('target')
    # # neg_features = neg_df
    # print(pos_df.shape)
    # print(neg_df.shape)
    # rng = np.random.default_rng()
    # resampled_neg_df = rng.choice(neg_df, len(pos_df), replace=False)
    # # print(neg_df[:, 14])
    # resampled_df = np.concatenate([pos_df, resampled_neg_df], axis=0)
    # resampled_labels = resampled_df.pop('target')
    # resampled_features = resampled_df
#%%
# def df_to_dataset(df, batch_size=32):
#     df = df.copy()
    
#     labels = np.array(df.pop('target'))
#     bool_labels = labels != 0 
#     features = np.array(df)

#     pos_features = features[bool_labels]
#     pos_labels = labels[bool_labels]

#     neg_features = features[~bool_labels]
#     neg_labels = labels[~bool_labels]

#     ids = np.arange(len(pos_features))
#     choices = np.random.choice(ids, len(neg_features))

#     resampled_pos_features = pos_features[choices]
#     resampled_pos_labels = pos_labels[choices]
#     print("resample pos:")
#     print(resampled_pos_features.shape)
#     print(resampled_pos_labels.shape)
#     print("resample neg:")
#     print(neg_features.shape)
#     print(neg_labels.shape)

#     resampled_features = np.concatenate([resampled_pos_features, neg_features], axis=0)
#     resampled_labels = np.concatenate([resampled_pos_labels, neg_labels], axis=0)
#     print("resample + concat:")
#     print(resampled_features.shape)
#     print(resampled_labels.shape)

#     resampled_features_df = pd.DataFrame(resampled_features, columns=df.columns)
#     resampled_labels_df = pd.DataFrame(resampled_labels, columns=['target'])

#     print("resample + concat + df:")
#     print(resampled_features_df.shape)
#     print(resampled_labels_df.shape)
#     print("FEATURESSSS")
#     print(resampled_features_df.head)
#     # print(resampled_features_df.shape)
#     print("LABELLSSS")
#     print(resampled_labels_df.head)
#     # print(resampled_labels_df.shape)

    # tf_dataset = tf.data.Dataset.from_tensor_slices(dict(resampled_features), resampled_labels)
    # tf_dataset = tf_dataset.shuffle(buffer_size=len(resampled_features)).batch(batch_size).prefetch(2)
    # print(len(tf_dataset))
    # return tf_dataset

#%%
def df_to_dataset(df, batch_size=32, resample=False):
    df = df.copy()
    if resample:
        pos_df = df[df['target'] == 1]
        neg_df = df[df['target'] == 0]
        pos_labels = pos_df.pop('target')
        pos_features = pos_df
        neg_labels = neg_df.pop('target')
        neg_features = neg_df
        # print('before resampling: ')
        # print(len(pos_features))
        # print(len(neg_features)) 
        pos_ds = tf.data.Dataset.from_tensor_slices((dict(pos_features), pos_labels))
        neg_ds = tf.data.Dataset.from_tensor_slices((dict(neg_features), neg_labels))
        
        resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        #resampled_ds = resampled_ds.apply(tf.data.experimental.assert_cardinality(54748))
        resampled_ds = resampled_ds.shuffle(buffer_size=len(df))
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        return resampled_ds.repeat(3)
    else:
        labels = df.pop('target')
        tf_dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
        return shuffled_tf_dataset.batch(batch_size).prefetch(2) # returning 32 samples per batch


#%%
# with large batch size
batch_size=1024
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
train_ds = df_to_dataset(df=train, batch_size=batch_size, resample=True)
val_ds = df_to_dataset(df=val, batch_size=batch_size)
test_ds = df_to_dataset(df=test, batch_size=batch_size)
# size = (len(train_ds)+len(val_ds)+len(test_ds))
# print('dataset sizes:')
# print(len(train_ds))
# print(len(val_ds))
# print(len(test_ds))

steps_per_epoch = np.ceil(2.0*pos/batch_size)
print(steps_per_epoch)
#%%
# train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(300000))
# val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(300000))
# test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(300000))

# to view 3 batches of data: 
# for person in train_dataset.take(3):
#     print(person[0]) #  the input dictionaries
#     print(person[1]) #  the binary categorical labels

# for unit in train.get('target'):
#     print(unit)

#%%
def get_normalization_layer(feature_name, dataset):
    # normalize numeric features
    normalizer = layers.Normalization(axis=None)
    # extract feature from dataset
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    normalizer.adapt(feature_ds)
    return normalizer
def get_category_encoding_layer(feature_name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)
    # extract feature from dataset
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    # 'learn' all possible feature values, assign each an int index 
    index.adapt(feature_ds)
    # encode integer index
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size(), output_mode="one_hot")
    # multi-hot encode indeices - lambda function captures layers
    return lambda feature: encoder(index(feature))

#%%
# testing pipeline steps with small batch size 
# batch_size=5

# #train, test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
# train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
# train_ds = df_to_dataset(train, batch_size=batch_size)
# val_ds = df_to_dataset(val, batch_size=batch_size)
# test_ds = df_to_dataset(test, batch_size=batch_size)
# print(len(train_ds))
# print(len(val_ds))
# print(len(test_ds))

# # view 2 batches
# [(train_features, label_batch)] = train_ds.take(1)
# print('Every feature:', list(train_features.keys()))
# print('A batch of ages:', train_features['ageGroup'])
# print('A batch of targets:', label_batch) 

# # normalization
# # bmi_col = train_features['bmi']
# # layer = get_normalization_layer('bmi', train_ds)
# # print(layer(bmi_col))
# # catgorization
# age_col = train_features['ageGroup']
# layer = get_category_encoding_layer('ageGroup', train_ds, 'string', max_tokens=14)
# print(age_col)
# print(layer(age_col))


#%%
# preprocess all features:
inputs = []
encoded_features =[]

# numerical
for header in ["bmi", "physicalHealth", "mentalHealth", 'sleepHours' ]:
    num_col = tf.keras.Input(shape=(1,), name=header)
    # keras inputs array
    inputs.append(num_col)

    norm_layer = get_normalization_layer(feature_name=header, dataset=train_ds)
    encoded_num_col = norm_layer(num_col)
    # encoded feature
    encoded_features.append(encoded_num_col)

# categorical
for header in ["smoking","alcoholDrinking","stroke","diffWalk",
                "sex", "ageGroup", "diabetic", "physicalActivity", 
                "overallHealth", "asthma", "kidneyDisease", "skinCancer"]:
    cat_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    # keras inputs array
    inputs.append(cat_col)

    cat_layer = get_category_encoding_layer(feature_name=header,
                                            dataset=train_ds, 
                                            dtype='string', 
                                            max_tokens=None)
    encoded_cat_col = cat_layer(cat_col)
    # encoded feature
    encoded_features.append(encoded_cat_col)
#%%
# merge list feature inputs into one vector
features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(units=128, activation="relu")(features)
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, output)
#%%
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
#%%
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics = ['accuracy'])

result = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=100,
                    steps_per_epoch=steps_per_epoch,
                    use_multiprocessing=True, 
                    verbose=2)

predictions = model.predict(test_ds)

binary_predictions = tf.round(predictions).numpy().flatten()

#%%
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()


#%%
print(classification_report(test.get('target'), binary_predictions))
# layer connectivity visualization
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# %%
