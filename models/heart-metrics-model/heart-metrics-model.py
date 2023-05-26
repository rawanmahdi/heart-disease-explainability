#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import shap
#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/models/heart-metrics-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe.drop_duplicates()
print(dataframe.shape)

#%%
feature_columns = []
# numeric columns - real values 
for column in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(column))

# bucketized 
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age,boundaries=[29, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator columns
dataframe["thal"] = dataframe['thal'].apply(str) 
thal = tf.feature_column.categorical_column_with_vocabulary_list(
    'thal', ['3', '6', '7'])
one_hot_thal = tf.feature_column.indicator_column(thal)
feature_columns.append(one_hot_thal)

dataframe["sex"] = dataframe['sex'].apply(str) 
sex = tf.feature_column.categorical_column_with_vocabulary_list(
    'sex', ['0', '1'])
one_hot_sex = tf.feature_column.indicator_column(sex)
feature_columns.append(one_hot_sex)

dataframe["cp"] = dataframe['cp'].apply(str) 
cp = tf.feature_column.categorical_column_with_vocabulary_list(
    'cp', ['0', '1', '2', '3'])
one_hot_cp = tf.feature_column.indicator_column(cp)
feature_columns.append(one_hot_cp)

dataframe["slope"] = dataframe['slope'].apply(str) 
slope = tf.feature_column.categorical_column_with_vocabulary_list(
    'slope', ['0', '1', '2'])
one_hot_slope = tf.feature_column.indicator_column(slope)
feature_columns.append(one_hot_slope)

# embedding column
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed column
age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
feature_columns.append(age_thal_crossed)

cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)
cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
feature_columns.append(cp_slope_crossed)

#%%
# function to build data pipeline to extract, shuffle and batch load the data
def create_dataset(df, batch_size=32):
    df = df.copy()
    labels = df.pop('target')
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size) # returning 32 samples per batch
#%%
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
train, test = train_test_split(dataframe, test_size=0.2, random_state=RANDOM_SEED)
ttest = test.copy()
ttrain = train.copy()
y_train = ttrain.pop('target')
X_train = ttrain
y_test = ttest.pop('target')
X_test = ttest
train_dataset = create_dataset(train)
test_dataset = create_dataset(test)
#%%
# to view 3 batches of data: 
# for person in train_dataset.take(3):
#     print(person[0]) #  the input dictionaries
#     print(person[1]) #  the binary categorical labels

# for unit in train.get('target'):
#     print(unit)
#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
#%%
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics = ['accuracy'])

history = model.fit(train_dataset, 
                    validation_data=test_dataset, 
                    batch_size=32,
                    epochs=500,
                    use_multiprocessing=True, 
                    verbose=1)
# history = model.fit(X_train, y_train,
#                     validation_data=(X_test, y_test), 
#                     batch_size=32,
#                     epochs=500,
#                     use_multiprocessing=True, 
#                     verbose=1)

#%%
model.evaluate(test_dataset)
#%%
predictions = model.predict(test_dataset)
print(predictions)
binary_predictions = tf.round(predictions).numpy().flatten()
print(classification_report(test.get('target'), binary_predictions))

#%%
sample =(test.iloc[204,:].shape)
# %%
#%%
def f(X):
    X_dict = dict(enumerate(X.flatten(), 1))
    predictions = model.predict(X_dict)
    return predictions.flatten()
#%%
import shap
explainer = shap.KernelExplainer(model=model, data=X_train.iloc[:50, :])
shap_values = explainer.shap_values(X=sample, nsamples='auto')

# %%

heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/models/heart-metrics-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
numer_labels = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
# y_train = train.pop('target')
# X_train = train[numer_labels]
# y_test = test.pop('target')
# X_test = test[numer_labels]
y_train = train.pop('target')
X_train = train
y_test = test.pop('target')
X_test = test
#%%
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train[numer_labels])
# X_train[numer_labels] = scaler.transform(X_train[numer_labels])
# X_test[numer_labels] = scaler.transform(X_test[numer_labels])

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%%
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])
model1.fit(X_train, y_train, 
          epochs=50,
          batch_size=32,
          validation_data=(X_test, y_test))
#%% 
model1.evaluate(X_test, y_test) 
#%%
explainer = shap.KernelExplainer(model1, X_train[:50, :])
#%%
shap.initjs()
shap_values = explainer.shap_values(X_train[30,:],nsamples='auto')
shap.force_plot(explainer.expected_value, shap_values[0], X_train[30,:])
