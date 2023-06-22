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
def load_df(path):
    df = pd.read_csv(path)
    df['target'] = np.where(df['heartDisease']=='Yes', 1, 0)
    df = df.drop(columns=['heartDisease', 'alcoholDrinking', 'skinCancer', 'kidneyDisease'])
    return df
df = load_df(r"C:\Users\Rawan Alamily\Downloads\McSCert Co-op\tabnet-heart\data\life-heart.csv")
#%%
yn = lambda x: 1 if x=='Yes' else 0
male = lambda x: 1 if x=='Male' else 0
def age(x):
    y = int(x[0:2])
    return y
def diabetes(x):
    if x=='Yes':
        y=1
    elif x=='No':
        y=0
    else:
        y=2
    return y
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
    return y
#%%

df.head()
#%%
def encode_strings(df):
    for i in range(df.iloc[:,:].shape[0]): 
        for j in range(1,3):
            df.iloc[i,j] = yn(df.iloc[i,j])
        df.iloc[i,5] = yn(df.iloc[i,5])
        df.iloc[i,6] = male(df.iloc[i,6])
        df.iloc[i,7] = age(df.iloc[i,7])
        df.iloc[i,8] = diabetes(df.iloc[i,8])
        df.iloc[i,9] = yn(df.iloc[i,9])
        df.iloc[i,10] = genHealth(df.iloc[i,10])
        df.iloc[i,12] = yn(df.iloc[i,12])

    df['smoking'] = df['smoking'].astype(str).astype(int)
    df['stroke'] = df['stroke'].astype(str).astype(int)
    df['diffWalk'] = df['diffWalk'].astype(str).astype(int)
    df['sex'] = df['sex'].astype(str).astype(int)
    df['ageGroup'] = df['ageGroup'].astype(str).astype(int)
    df['diabetic'] = df['diabetic'].astype(str).astype(int)
    df['physicalActivity'] = df['physicalActivity'].astype(str).astype(int)
    df['overallHealth'] = df['overallHealth'].astype(str).astype(int)
    df['asthma'] = df['asthma'].astype(str).astype(int)
    return df 
#%%
df = df.iloc[:25000,:]
df = encode_strings(df)    
#%%
df = df.copy()

#%%
def split_sample(df):
    dff = df.copy()
    y = dff.pop('target')
    X = dff

    train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.9*len(df))])
    y_train = train.pop('target')
    X_train = train
    y_val = val.pop('target')
    X_val = val
    y_test = test.pop('target')
    X_test = test

    rus = RandomUnderSampler(random_state=0)
    rus.fit(X,y)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)
    X_val_resampled, y_val_resampled= rus.fit_resample(X_val, y_val)

    return X_train_resampled, y_train_resampled, X_val_resampled, y_val_resampled, X_test, y_test

#%%
# scaler = StandardScaler().fit(X_train_resampled)
# X_train = scaler.transform(X_train_resampled)
# X_val = scaler.transform(X_val_resampled)
# X_test = scaler.transform(X_test)
#%%
def df_to_dataset(features, labels, batch_size=32):
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).cache()
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size).prefetch(2)# returning 32 samples per batch
#%%
X_train, y_train, X_val, y_val, X_test, y_test = split_sample(df)
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
for header in ["smoking","stroke","diffWalk",
                "sex", "ageGroup", "diabetic", "physicalActivity", 
                "overallHealth", "asthma"]:
    
    # declare header as a keras Input
    cat_col = tf.keras.Input(shape=(1,), name=header)
    # keras inputs array
    inputs.append(cat_col)

    # get preprocessing layer 
    cat_layer = get_category_encoding_layer(feature_name=header,
                                            dataset=train_ds, 
                                            dtype='int', 
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

# %%
