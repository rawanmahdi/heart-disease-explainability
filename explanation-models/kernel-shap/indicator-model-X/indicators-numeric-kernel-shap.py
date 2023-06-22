#%%
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report 
from imblearn.under_sampling import RandomUnderSampler
import shap
import tensorflow as tf
from tensorflow import keras
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
def predict_fn(X):
    bmi = tf.convert_to_tensor(X[:,0],dtype=np.float64) #float
    smoking = tf.convert_to_tensor(X[:,1],dtype=np.int64) #str
    stroke = tf.convert_to_tensor(X[:,2],dtype=np.int64) #str
    physical = tf.convert_to_tensor(X[:,3],dtype=np.int64) #int
    mental = tf.convert_to_tensor(X[:,4],dtype=np.int64) #int
    walk = tf.convert_to_tensor(X[:,5],dtype=np.int64) #str
    sex = tf.convert_to_tensor(X[:,6],dtype=np.int64) #str
    age = tf.convert_to_tensor(X[:,7],dtype=np.int64) #str
    diabetic = tf.convert_to_tensor(X[:,8],dtype=np.int64) #str
    activity = tf.convert_to_tensor(X[:,9],dtype=np.int64) #str
    health = tf.convert_to_tensor(X[:,10],dtype=np.int64) #str
    sleep = tf.convert_to_tensor(X[:,11],dtype=np.int64)
    asthma = tf.convert_to_tensor(X[:,12],dtype=np.int64) #str
    X_dict = {'bmi': bmi, 'smoking': smoking, 'stroke': stroke, 'physicalHealth': physical,
             'mentalHealth': mental, 'diffWalk': walk, 'sex': sex, 'ageGroup': age,'diabetic': diabetic, 'physicalActivity': activity, 
             'overallHealth': health, 'sleepHours': sleep, 'asthma': asthma}
    X_ds = tf.data.Dataset.from_tensor_slices((X_dict))
    X_ds = X_ds.batch(128)
    return model.predict(X_ds)

#%%
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/normalized-model")
#%%
explainer = shap.KernelExplainer(predict_fn, data=X_train)
#%%
# get single input shap feature plot
shap_values = explainer.shap_values(X_test.iloc[1,:])
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[1,:])
#%%
# get for multiple input
shap_values = explainer.shap_values(X_test.iloc[:100,:]);
print("done finding shap values")
#%%
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[:200])

# %%
# bar plot for multiple inputs
shap.summary_plot(shap_values, X_test, plot_type="bar")
# %%
