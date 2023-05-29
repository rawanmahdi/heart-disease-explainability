#%%
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import shap

#%%
# import data
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/models/heart-metrics-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)

train, val, test = np.split(dataframe.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])

# preprocess
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test

scaler = StandardScaler().fit(X_train)
X_train_NORM = scaler.transform(X_train)
X_val_NORM = scaler.transform(X_val)
X_test_NORM = scaler.transform(X_test)
#%%
# load saved model
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/models/heart-metrics-model/saved-model")
#%%
# fit kernel explainer using training data
explainer = shap.KernelExplainer(model, X_train_NORM[:500, :])
shap.initjs()
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
