
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
train, val, test = np.split(dataframe.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test
print("Training sample:",len(y_train))
print("Testing sample:", len(y_test))
print("Validation sample:",len(y_val))
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
X_val_resampled, y_val_resampled = rus.fit_resample(X_val, y_val)
neg0, pos0 = np.bincount(y_train_resampled)
print("No.negative samples after undersampling",neg0)
print("No.positive samples after undersampling",pos0)
print(len(y_train_resampled))
print(len(y_val_resampled))
#%%
def df_to_dataset(features, labels, batch_size=512):
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).cache()
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size).prefetch(2)# returning 32 samples per batch

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
train_resampled_ds= df_to_dataset(X_train_resampled, y_train_resampled)
#val_ds= df_to_dataset(X_val_resampled, y_val_resampled)
val_ds = df_to_dataset(X_val, y_val)
test_ds= df_to_dataset(X_test, y_test)

#%%
for i in train_resampled_ds.take(1):
    print(i)

#%%
inputs = []
encoded_features =[]

# numerical
for header in ["bmi", "physicalHealth", "mentalHealth", 'sleepHours' ]:
    num_col = tf.keras.Input(shape=(1,), name=header)
    # keras inputs array
    inputs.append(num_col)

    norm_layer = get_normalization_layer(feature_name=header, dataset=train_resampled_ds)
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
                                            dataset=train_resampled_ds, 
                                            dtype='string', 
                                            max_tokens=None)
    encoded_cat_col = cat_layer(cat_col)
    # encoded feature
    encoded_features.append(encoded_cat_col)
# %%
# KERAS FUNCTIONAL API - MODEL BUILD   
# merge list feature inputs into one vector
features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(units=64, activation="relu")(features)
x = tf.keras.layers.Dropout(rate=0.4)(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.4)(x)
x = tf.keras.layers.Dense(units=128, activation="relu")(x)
x = tf.keras.layers.Dropout(rate=0.4)(x)
x = tf.keras.layers.Dense(units=64, activation='relu')(x)
output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, output)
#%%
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics = ['accuracy'])

result = model.fit(
    # X_train_resampled,y_train_resampled,
                    train_resampled_ds,
                    validation_data=val_ds, 
                    # validation_data=(X_val, y_val),
                    epochs=100,
                    verbose=1)

# %%
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
#%%
print(test_ds.take(1))
#%%
predictions = model.predict(test_ds)
binary_predictions = tf.round(predictions).numpy().flatten()
print(classification_report(y_test, binary_predictions))
# %%
model.save("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/saved-model")
#%%
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/saved-model")

#%%
for i in X_train_resampled.iloc[0, :]:
    print(type(i))
#%%
X_train_resampled = X_train_resampled.astype({'bmi':np.float32,
                                              'smoking':'string',
                                              'alcoholDrinking':'string',
                                              'stroke':'string',
                                              'physicalHealth':np.int32,
                                              'mentalHealth':np.int32,
                                              'diffWalk':'string',
                                              'sex':'string',
                                              'ageGroup':'string',
                                              'diabetic':'string',
                                              'physicalActivity':'string',
                                              'overallHealth':'string',
                                              'sleepHours':np.int32,
                                              'asthma':'string',
                                              'kidneyDisease':'string',
                                              'skinCancer':'string'})
#%%
for i in X_train_resampled.iloc[0, :]:
    print(type(i))
#%%
tensors = (dict(X_train_resampled))
ds = tf.data.Dataset.from_tensor_slices(tensors=tensors).batch(512)

model.predict(ds)
#%%
import shap
data = X_train_resampled.iloc[:10,:]
data.shape[1]
#%%
print(data.iloc[:3,:])
#%%
for i in data.iloc[1, :]:
    print(type(i))
#%%
def f(X):
    ## FIX TO USE ACTUAL X DATA
    bmi = tf.convert_to_tensor(X[:,0],dtype=np.float64) #float
    smoking = tf.convert_to_tensor(X[:,1]) #str
    alcohol = tf.convert_to_tensor(X[:,2]) #str
    stroke = tf.convert_to_tensor(X[:,3]) #str
    physical = tf.convert_to_tensor(X[:,4],dtype=np.int64) #int
    mental = tf.convert_to_tensor(X[:,5],dtype=np.int64) #int
    walk = tf.convert_to_tensor(X[:,6]) #str
    sex = tf.convert_to_tensor(X[:,7]) #str
    age = tf.convert_to_tensor(X[:,8]) #str
    diabetic = tf.convert_to_tensor(X[:,9]) #str
    activity = tf.convert_to_tensor(X[:,10]) #str
    health = tf.convert_to_tensor(X[:,11]) #str
    sleep = tf.convert_to_tensor(X[:,12],dtype=np.int64) #int
    asthma = tf.convert_to_tensor(X[:,13]) #str
    kidney = tf.convert_to_tensor(X[:,14]) #str
    skinCancer = tf.convert_to_tensor(X[:,15]) #str
    dict = {'bmi': bmi, 'smoking': smoking,'alcoholDrinking': alcohol, 'stroke': stroke, 'physicalHealth': physical,
             'mentalHealth': mental, 'diffWalk': walk, 'sex': sex, 'ageGroup': age,'diabetic': diabetic, 'physicalActivity': activity, 
             'overallHealth': health, 'sleepHours': sleep, 'asthma': asthma,  'kidneyDisease': kidney, 'skinCancer': skinCancer}
    # X_ds = tf.data.Dataset.from_tensor_slices((bmi, smoking, alcohol,stroke, 
    #                                      physical,mental,walk,sex,age,
    #                                      diabetic,activity,health,sleep,
    #                                      asthma,kidney,skinCancer))
    X_ds = tf.data.Dataset.from_tensor_slices((dict))
    X_ds = X_ds.batch(128)
    print(X_ds.take(1))
    # tensors = (dict(data))
    # X_ds = tf.data.Dataset.from_tensor_slices(tensors=tensors).batch(512)
    return model.predict(X_ds)

explainer = shap.KernelExplainer(f, data=data)
shap.initjs()
# %%
# test on various individual samples using test dataset #%%
shap_values = explainer.shap_values(X_test.iloc[1,:])
#%%
shap.force_plot(explainer.expected_value, shap_values[0], X_test)

# %%
shap.force_plot(explainer.expected_value, shap_values[0], X_test,type='bar')

# %%
shap.summary_plot(shap_values, X_test, plot_type="bar")

