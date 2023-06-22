#%%
from shaprpy import explain
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
dataframe['Sex'] = np.where(dataframe['Sex'] == "M",1,0)
dataframe['ExerciseAngina'] = np.where(dataframe['ExerciseAngina'] == "Y",1,0)
def cp(x):
    if x=='TA':
        y=1
    elif x=='ATA':
        y=0
    elif x=='NAP':
        y=2
    else:
        y=3
    return np.int32(y)

def restECG(x):
    if x=='Normal':
        y=1
    elif x=='ST':
        y=0
    else:
        y=2
    return np.int32(y)

def slope(x):
    if x=='Up':
        y=1
    elif x=='Flat':
        y=0
    else:
        y=2
    return np.int32(y)
for i in range(dataframe.iloc[:, 2].shape[0]):
    dataframe.iloc[i,2] = cp(dataframe.iloc[i,2])
for i in range(dataframe.iloc[:, 6].shape[0]):
    dataframe.iloc[i,6] = restECG(dataframe.iloc[i,6])
for i in range(dataframe.iloc[:, 10].shape[0]):
    dataframe.iloc[i,10] = slope(dataframe.iloc[i,10])
#%%
dataframe.head(10)
#%%
train, val, test = np.split(dataframe.sample(frac=1), [int(0.5*len(dataframe)), int(0.9*len(dataframe))])
# train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
y_train = train.pop('HeartDisease')
columns = train.columns
print(columns)
X_train = train
y_val = val.pop('HeartDisease')
X_val = val
y_test = test.pop('HeartDisease')
X_test = test

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_NORM = scaler.transform(X_train)
X_val_NORM = scaler.transform(X_val)
X_test_NORM = scaler.transform(X_test)

#%%
dfX_train = pd.DataFrame(X_train_NORM, columns=columns)
dfX_test = pd.DataFrame(X_test_NORM, columns=columns)
#%%
dfX_test.dtypes
#%%
# load saved model
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-model")
#%%
def predict_fn(m,x):
    pred = m.predict(x)
    return pred.reshape(pred.shape[0],)
#%%
df_shapley, pred_explain, internal, timing = explain(
    model = model,
    x_train = dfX_train,
    x_explain = dfX_test,
    approach = 'empirical',
    prediction_zero = y_train.mean().item(),
)

#%%
print(df_shapley)
#%%
print(pred_explain)
#%%
import matplotlib.pyplot as plt 
#%%
#%%
df_shapley = df_shapley.iloc[:,1:]
pred = pred_explain[1]
#.plot.bar()
#%%
print(df_shapley)
# %%
df_shapley.plot.bar()
#%%
print(pred)
# %%
mean_abs_shap = {}
for col in df_shapley.columns:
    mean_abs_shap[col] = np.mean(np.absolute(df_shapley[col]))

print(mean_abs_shap)
#%%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
features = mean_abs_shap.keys()
means = mean_abs_shap.values()
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
ax.bar(features,means)
plt.show()
# %%
sample = df_shapley.iloc[1]
# %%
print(sample)
sample_p = pred_explain[1]
print(sample_p)
# %%
print(test.iloc[1])
# %%
