from shaprpy import explain
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/life-heart.csv'
dataframe = pd.read_csv(heart_csv_path)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
df = dataframe.copy()
# designate for fitting rus
y = df.pop('target')
X = df
train, val, test = np.split(dataframe.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])
y_train = train.pop('target')
X_train = train
y_test = test.pop('target')
X_test = test
# resample via undersampling majority class - this is favoured over oversampling as the dataset is very large
rus = RandomUnderSampler(random_state=0)
rus.fit(X,y)
# only resample training dataset
X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)
neg0, pos0 = np.bincount(y_train_resampled)
print("No.negative samples after undersampling",neg0)
print("No.positive samples after undersampling",pos0)
#%%
encode = lambda x: 1 if x=='Yes' else 0
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
print(X.shape)
#%%
X_norm = X.copy()
X_norm[0,1] = encode(X_norm[0,1])
X_norm[0,2] = encode(X_norm[0,2])
X_norm[0,3] = encode(X_norm[0,3])
X_norm[0,6] = encode(X_norm[0,6])
X_norm[0,7] = male(X_norm[0,7])
X_norm[0,8] = age(X_norm[0,8])
X_norm[0,9] = diabetes(X_norm[0,9])
X_norm[0,10] = encode(X_norm[0,10])
X_norm[0,11] = genHealth(X_norm[0,11])
X_norm[0,13] = encode(X_norm[0,13])
X_norm[0,14] = encode(X_norm[0,14])
X_norm[0,15] = encode(X_norm[0,15])
#%%
def encode_strings(arr):
    for row in arr:
        row[1] = encode(row[1])
        row[2] = encode(row[2])
        row[3] = encode(row[3])
        row[6] = encode(row[6])
        row[7] = male(row[7])
        row[8] = age(row[8])
        row[9] = diabetes(row[9])
        row[10] = encode(row[10])
        row[11] = genHealth(row[11])
        row[13] = encode(row[13])
        row[14] = encode(row[14])
        row[15] = encode(row[15])
    return arr

#%%
scaler = StandardScaler().fit(X_train_resampled)
X_train_NORM = scaler.transform(X_train_resampled)
X_test_NORM = scaler.transform(X_test)
#%%
dfX_train = pd.DataFrame(X_train_NORM, columns=dataframe.columns)
dfX_test = pd.DataFrame(X_test_NORM, columns=dataframe.columns)
#%%
dfX_test.dtypes
#%%
# load saved model
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-model")
#%%
def f(X):
    bmi = tf.convert_to_tensor(X[:,0],dtype=np.float64) #float
    smoking = tf.convert_to_tensor(X[:,1]) #str
    stroke = tf.convert_to_tensor(X[:,2]) #str
    physical = tf.convert_to_tensor(X[:,3],dtype=np.int64) #int
    mental = tf.convert_to_tensor(X[:,4],dtype=np.int64) #int
    walk = tf.convert_to_tensor(X[:,5]) #str
    sex = tf.convert_to_tensor(X[:,6]) #str
    age = tf.convert_to_tensor(X[:,7]) #str
    diabetic = tf.convert_to_tensor(X[:,8]) #str
    activity = tf.convert_to_tensor(X[:,9]) #str
    health = tf.convert_to_tensor(X[:,10]) #str
    asthma = tf.convert_to_tensor(X[:,11]) #str
    bdsa = tf.convert_to_tensor(X[:,12],dtype=np.float64)
    X_dict = {'bmi': bmi, 'smoking': smoking, 'stroke': stroke, 'physicalHealth': physical,
             'mentalHealth': mental, 'diffWalk': walk, 'sex': sex, 'ageGroup': age,'diabetic': diabetic, 'physicalActivity': activity, 
             'overallHealth': health, 'asthma': asthma, 'BDSA' : bdsa}
    X_ds = tf.data.Dataset.from_tensor_slices((X_dict))
    X_ds = X_ds.batch(128)
    return model.predict(X_ds)
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