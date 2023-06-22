#%%
from shaprpy import explain
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
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
df = df.iloc[:1000,:]
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
X_train, y_train, X_val, y_val, X_test, y_test = split_sample(df)
#%%
def predict_fn(model, X):
    bmi = tf.convert_to_tensor(X.iloc[:,0],dtype=np.float64) #float
    smoking = tf.convert_to_tensor(X.iloc[:,1],dtype=np.int64) #str
    stroke = tf.convert_to_tensor(X.iloc[:,2],dtype=np.int64) #str
    physical = tf.convert_to_tensor(X.iloc[:,3],dtype=np.int64) #int
    mental = tf.convert_to_tensor(X.iloc[:,4],dtype=np.int64) #int
    walk = tf.convert_to_tensor(X.iloc[:,5],dtype=np.int64) #str
    sex = tf.convert_to_tensor(X.iloc[:,6],dtype=np.int64) #str
    age = tf.convert_to_tensor(X.iloc[:,7],dtype=np.int64) #str
    diabetic = tf.convert_to_tensor(X.iloc[:,8],dtype=np.int64) #str
    activity = tf.convert_to_tensor(X.iloc[:,9],dtype=np.int64) #str
    health = tf.convert_to_tensor(X.iloc[:,10],dtype=np.int64) #str
    sleep = tf.convert_to_tensor(X.iloc[:,11],dtype=np.int64)
    asthma = tf.convert_to_tensor(X.iloc[:,12],dtype=np.int64) #str
    X_dict = {'bmi': bmi, 'smoking': smoking, 'stroke': stroke, 'physicalHealth': physical,
             'mentalHealth': mental, 'diffWalk': walk, 'sex': sex, 'ageGroup': age,'diabetic': diabetic, 'physicalActivity': activity, 
             'overallHealth': health, 'sleepHours': sleep, 'asthma': asthma}
    X_ds = tf.data.Dataset.from_tensor_slices((X_dict))
    X_ds = X_ds.batch(128)
    pred =  model.predict(X_ds)
    return pred.reshape(pred.shape[0])
#%%
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/normalized-model")

#%%
df_shapley, pred_explain, internal, timing = explain(
    model = model,
    x_train = X_train,
    x_explain = X_test,
    approach = 'empirical',
    predict_model=predict_fn,
    prediction_zero = y_train.mean().item(), 
    n_batches=50
)
  
#%%
print(df_shapley)
print(pred_explain)

#%%
df_shapley = df_shapley.iloc[:,1:]
pred = pred_explain[1]
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
sorted_means = dict(sorted(mean_abs_shap.items(), key=lambda x:x[1], reverse=True))
print(sorted_means)
#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

features = sorted_means.keys()
means = sorted_means.values()
plt.setp(ax.get_yticklabels(), fontsize=10, rotation='vertical')
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
ax.bar(features,means)
plt.show()
# %%
# PLOT SINGLE INDIVIDUAL
sample = df_shapley.iloc[2,:]
sample_features = (X_test.iloc[2,:])
sample_pred = pred_explain[2]
print(sample, sample_features, sample_pred)
# %%
sample = sample.sort_values()
sample.plot.barh(stacked=True)
print("Feature Values: \n",sample_features)
print("Prediction: ", sample_pred)
#%%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
features = sorted_means.keys()
means = sorted_means.values()
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
ax.bar(features,means)
plt.show()
#%%

#%%
##
labels = ['person no.1']
data = sample
data_cum = data.cumsum()
category_colors = plt.colormaps['RdYlGn'](
    np.linspace(0.15, 0.85, data.shape[0]))

fig, ax = plt.subplots(figsize=(9.2, 5))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.set_xlim(0, np.sum(data).max())

for i,(colname, color) in enumerate(zip(sample_features, category_colors)):
    widths = data[i]
    starts = data_cum[i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=color)

    r, g, b, _ = color
    text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
    ax.bar_label(rects, label_type='center', color=text_color)
ax.legend(ncols=len(sample_features), bbox_to_anchor=(0, 1),
            loc='lower left', fontsize='small')
# %%
