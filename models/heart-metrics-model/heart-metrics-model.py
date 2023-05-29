#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report 
import shap

# %%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/models/heart-metrics-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
train, val, test = np.split(dataframe.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])
# train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test
print(len(X_train))
print(len(X_val))
print(len(X_test))
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_NORM = scaler.transform(X_train)
X_val_NORM = scaler.transform(X_val)
X_test_NORM = scaler.transform(X_test)

#%%
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
model1.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])
#%%
model1.fit(X_train_NORM, y_train, 
          epochs=70,
          batch_size=32,
          validation_data=(X_val_NORM, y_val))
#%% 
model1.evaluate(X_test_NORM, y_test) 
#%%
# fit kernel explainer using training data
explainer = shap.KernelExplainer(model1, X_train_NORM[:500, :])

# %%
# test on various individual samples using test dataset 
shap_values = explainer.shap_values(X_test_NORM[1,:],nsamples='auto')
shap.force_plot(explainer.expected_value, shap_values[0], X_test_NORM[1,:])

# %%
i=0
for label in dataframe.columns:
    if label != 'target':
        print(label,";",shap_values[0][i])
        i+=1
print(X_test.iloc[1,:])
print(y_test.iloc[1])

# %%
shap_values = explainer.shap_values(X_test_NORM[2,:],nsamples='auto')
shap.force_plot(explainer.expected_value, shap_values[0], X_test_NORM[2,:])

# %%
i=0
for label in dataframe.columns:
    if label != 'target':
        print(label,";",shap_values[0][i])
        i+=1
print(X_test.iloc[2,:])
print(y_test.iloc[2])

# %%
shap_values = explainer.shap_values(X_test_NORM[3,:],nsamples='auto')
shap.force_plot(explainer.expected_value, shap_values[0], X_test_NORM[3,:])

# %%
i=0
for label in dataframe.columns:
    if label != 'target':
        print(label,";",shap_values[0][i])
        i+=1
print(X_test.iloc[3,:])
print(y_test.iloc[3])
#%%
shap_values = explainer.shap_values(X_test_NORM[5,:],nsamples='auto')
shap.force_plot(explainer.expected_value, shap_values[0], X_test_NORM[4,:])
# %%
i=0
for label in dataframe.columns:
    if label != 'target':
        print(label,";",shap_values[0][i])
        i+=1
print(X_test.iloc[5,:])
print(y_test.iloc[5])
