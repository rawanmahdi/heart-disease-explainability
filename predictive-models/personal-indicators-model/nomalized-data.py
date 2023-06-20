#%%
import tensorflow as tf
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report 
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from keras import regularizers
#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/life-heart.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
dataframe = dataframe.iloc[1:25000, :]
df = dataframe.copy()
df.head()
#%%
#FEATURE ENCODING
encode = lambda x: np.int64(1) if x=='Yes' else np.int64(0)
male = lambda x: np.int64(1) if x=='Male' else np.int64(0)
typeconv = lambda x: np.int64(x)
def age(x):
    y = int(str(x)[0:2])
    return np.float64(y)
def diabetes(x):
    if x=='Yes':
        y=1
    elif x=='No':
        y=0
    else:
        y=2
    return np.float64(y)
def genHealth(x):
    if x=='Very good':
        y=0
    elif x=='Good':
        y=1
    elif x=='Excellent':
        y=2
    elif x=='Fair':
        y=3
    else:
        y=4
    return np.float64(y)
#%%
for i in range(df.iloc[:, 1].shape[0]):
    df.iloc[i,1] = encode(df.iloc[i,1])
for i in range(df.iloc[:, 2].shape[0]):
    df.iloc[i,2] = encode(df.iloc[i,2])
for i in range(df.iloc[:, 3].shape[0]):
    df.iloc[i,3] = encode(df.iloc[i,3])
for i in range(df.iloc[:, 4].shape[0]):
    df.iloc[i,4] = np.float64(df.iloc[i,4])
for i in range(df.iloc[:, 5].shape[0]):
    df.iloc[i,5] = np.float64(df.iloc[i,5])
for i in range(df.iloc[:, 6].shape[0]):
    df.iloc[i,6] = encode(df.iloc[i,6]) 
for i in range(df.iloc[:, 7].shape[0]):
    df.iloc[i,7] = male(df.iloc[i,7])
for i in range(df.iloc[:, 8].shape[0]):
    df.iloc[i,8] = age(df.iloc[i,8])
for i in range(df.iloc[:, 9].shape[0]):
    df.iloc[i,9] = diabetes(df.iloc[i,9])
for i in range(df.iloc[:, 10].shape[0]):
    df.iloc[i,10] = encode(df.iloc[i,10])
for i in range(df.iloc[:, 11].shape[0]):
    df.iloc[i,11] = genHealth(df.iloc[i,11])
for i in range(df.iloc[:, 12].shape[0]):
    df.iloc[i,12] = np.float64(df.iloc[i,12])
for i in range(df.iloc[:, 13].shape[0]):
    df.iloc[i,13] = encode(df.iloc[i,13]) 
for i in range(df.iloc[:, 14].shape[0]):
    df.iloc[i,14] = encode(df.iloc[i,14])  
for i in range(df.iloc[:, 15].shape[0]):
    df.iloc[i,15] = encode(df.iloc[i,15]) 
df.head()
#%%
def split_process(df):
    # SPLIT 
    train, val, test = np.split(df.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])
    y_train = train.pop('target')
    X_train = train
    y_val = val.pop('target')
    X_val = val
    y_test = test.pop('target')
    X_test = test
    # RESAMPLE 
    rus = RandomUnderSampler(random_state=0)
    y = df.pop('target')
    X = df
    rus.fit(X,y)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)
    neg0, pos0 = np.bincount(y_train_resampled)
    X_val_res, y_val_res = rus.fit_resample(X_val, y_val)
    print("No.negative samples after undersampling",neg0)
    print("No.positive samples after undersampling",pos0)
    # SCALE
    # scaler = StandardScaler().fit(X_train_resampled)
    # X_train_NORM = scaler.transform(X_train_resampled)
    # X_val_NORM = scaler.transform(X_val_res)
    # X_test_NORM = scaler.transform(X_test)

    # return X_train_NORM, y_train_resampled, X_test_NORM, y_test, X_val_NORM, y_val_res
    # return X_train_resampled, y_train_resampled, X_test, y_test, X_val_res, y_val_res
    return X_train, y_train, X_test, y_test, X_val, y_val
#%%
X_train, y_train, X_test, y_test, X_val, y_val = split_process(df)
#%%
type(X_train)
#%%
for thing in X_train.iloc[1,:]:
    print(type(thing))
#%%
def df_to_dataset(features, labels, batch_size=32):
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).cache()
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size).prefetch(2)# returning 32 samples per batch
#%%
train_ds= df_to_dataset(X_train, y_train)
val_ds = df_to_dataset(X_val, y_val)
test_ds= df_to_dataset(X_test, y_test)
#%%
def get_normalization_layer(feature_name, dataset, batch_size=32):
    # normalize numeric features
    normalizer = layers.Normalization(axis=None)
    # extract feature from dataset
    #feature_data = dataset[feature_name]
    feature_data = dataset.map(lambda x, y: x[feature_name])
    normalizer.adapt(feature_data, batch_size=batch_size)
    return normalizer
def get_category_encoding_layer(feature_name, dataset, dtype, max_tokens=None, batch_size=32):
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)
    # extract feature from dataset
    #feature_ds = dataset[feature_name]
    feature_data = dataset.map(lambda x, y: x[feature_name])
    # 'learn' all possible feature values, assign each an int index 
    index.adapt(feature_data, batch_size=batch_size)
    # encode integer index
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size(), output_mode="one_hot")
    # multi-hot encode indeices - lambda function captures layers
    return lambda feature: encoder(index(feature))

#%%
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
    
    # declare header as a keras Input
    cat_col = tf.keras.Input(shape=(1,), name=header, dtype='int')
    # keras inputs array
    inputs.append(cat_col)

    # get preprocessing layer 
    cat_layer = get_category_encoding_layer(feature_name=header,
                                            dataset=train_ds, 
                                            dtype='string', 
                                            max_tokens=None)
    encoded_cat_col = cat_layer(cat_col)
    # encoded feature
    encoded_features.append(encoded_cat_col)
# %%
# KERAS FUNCTIONAL API - MODEL BUILD   
# merge list feature inputs into one vector
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
features = layers.concatenate(encoded_features)
x = layers.Dense(
    units=64,
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)
)(features)
x = layers.Dense(units=90, activation='relu')(x)
x = layers.Dropout(rate=0.67)(x)
x = layers.Dense(units=128, activation="relu")(x)
x = layers.Dropout(rate=0.7)(x)
x = layers.Dense(units=64, activation='relu')(x)
output = layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, output)
#%%
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics = ['accuracy'])
#%%
result = model.fit(
    # X_train_resampled,y_train_resampled,
                    train_ds,
                    validation_data=val_ds, 
                    # validation_data=(X_val, y_val),
                    epochs=100,
                    verbose=1)

# %%
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
#%%
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()
#%%
predictions = model.predict(test_ds)
binary_predictions = tf.round(predictions).numpy().flatten()
print(classification_report(y_test, binary_predictions))
# %%
model.save("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/normalized-model")
