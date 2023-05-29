
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
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/models/lifestyle-metrics-model/data/life-heart.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
neg, pos = np.bincount(dataframe['target'])


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
        pos_ds = tf.data.Dataset.from_tensor_slices((dict(pos_features), pos_labels))
        neg_ds = tf.data.Dataset.from_tensor_slices((dict(neg_features), neg_labels))
        
        resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.65, 0.35])
        #resampled_ds = resampled_ds.apply(tf.data.experimental.assert_cardinality(54748))
        resampled_ds = resampled_ds.shuffle(buffer_size=len(df))
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2).repeat()
        return resampled_ds
    else:
        labels = df.pop('target')
        tf_dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels)).cache()
        shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
        return tf_dataset.batch(batch_size).prefetch(2)# returning 32 samples per batch

#%%
# with large batch size
batch_size=64
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
dataframe = dataframe.sample(frac=1)
#train, val = train_test_split(dataframe, test_size=0.3, random_state=RANDOM_SEED)
resampled_train_ds = df_to_dataset(df=train, batch_size=batch_size, resample=True)
val_ds = df_to_dataset(df=val, batch_size=batch_size, resample=False)
test_ds = df_to_dataset(df=test, batch_size=batch_size)
steps_per_epoch = np.ceil(2.0*pos/batch_size)
print(steps_per_epoch)
#%%
def get_normalization_layer(feature_name, dataset):
    # normalize numeric features
    normalizer = layers.Normalization(axis=None)
    # extract feature from dataset
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    normalizer.adapt(feature_ds, batch_size=batch_size, steps=steps_per_epoch)
    return normalizer
def get_category_encoding_layer(feature_name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)
    # extract feature from dataset
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    # 'learn' all possible feature values, assign each an int index 
    index.adapt(feature_ds, batch_size=batch_size, steps=steps_per_epoch)
    # encode integer index
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size(), output_mode="one_hot")
    # multi-hot encode indeices - lambda function captures layers
    return lambda feature: encoder(index(feature))

#%%
# preprocess all features:
inputs = []
encoded_features =[]

# numerical
for header in ["bmi", "physicalHealth", "mentalHealth", 'sleepHours' ]:
    num_col = tf.keras.Input(shape=(1,), name=header)
    # keras inputs array
    inputs.append(num_col)

    norm_layer = get_normalization_layer(feature_name=header, dataset=resampled_train_ds)
    encoded_num_col = norm_layer(num_col)
    # encoded feature
    encoded_features.append(encoded_num_col)

# categorical
for header in ["smoking","alcoholDrinking","stroke","diffWalk",
                "sex", "ageGroup", "diabetic", "physicalActivity", 
                "overallHealth", "asthma", "kidneyDisease", "skinCancer"]:
    
    # declare header as a keras Input
    cat_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    # keras inputs array
    inputs.append(cat_col)

    # get preprocessing layer 
    cat_layer = get_category_encoding_layer(feature_name=header,
                                            dataset=resampled_train_ds, 
                                            dtype='string', 
                                            max_tokens=None)
    encoded_cat_col = cat_layer(cat_col)
    # encoded feature
    encoded_features.append(encoded_cat_col)
#%%
# KERAS FUNCTIONAL API - MODEL BUILD   
# merge list feature inputs into one vector
features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(units=128, activation="relu")(features)
x = tf.keras.layers.Dropout(rate=0.2)(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, output)

#%%
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics = ['accuracy'])

result = model.fit(resampled_train_ds, 
                    validation_data=val_ds, 
                    epochs=20,
                    steps_per_epoch=steps_per_epoch,
                    use_multiprocessing=True, 
                    verbose=1)


#%%
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()


#%%
predictions = model.predict(test_ds)
binary_predictions = tf.round(predictions).numpy().flatten()
print(classification_report(test.get('target'), binary_predictions))
#%%
# layer connectivity visualization
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


#%%
df = pd.read_csv(heart_csv_path)
df['target'] = np.where(df['heartDisease']=='Yes', 1, 0)
df = df.drop(columns=['heartDisease'])
pos_df = df[df['target'] == 1]
neg_df = df[df['target'] == 0]
pos_labels = pos_df.pop('target')
pos_features = pos_df
neg_labels = neg_df.pop('target')
neg_features = neg_df
pos_ds = tf.data.Dataset.from_tensor_slices((dict(pos_features), pos_labels))
neg_ds = tf.data.Dataset.from_tensor_slices((dict(neg_features), neg_labels))
# sampled_person = pos_ds.take(1)
# prediction = model.predict(sampled_person)
# binary_predictions = tf.round(predictions).numpy().flatten()
# print(binary_predictions)
# %%
