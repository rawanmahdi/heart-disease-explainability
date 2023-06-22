#%%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from aix360.algorithms.protodash import ProtodashExplainer

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
def df_to_dataset(features, labels, batch_size=512):
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).cache()
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size).prefetch(2)# returning 32 samples per batch

# train = X_train_resampled
# train['target'] = y_train_resampled
train = df_to_dataset(X_train_resampled, y_train_resampled)
test = df_to_dataset(X_test, y_test)
#%%
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/saved-model")
#%%
p_test = tf.round(model.predict(train))
print(p_test.shape,X_train_resampled.shape)
#%%
print(p_test)
#%%
z_test = np.hstack((X_train_resampled,p_test))
z_test_pos = z_test[z_test[:,-1]==1,:]
print(z_test_pos.shape)
#%%
sample_id = 114
sample = X_test.iloc[sample_id,:]
print(y_test.iloc[sample_id])
print(type(sample))
print(sample.shape)
#%%
bmi = tf.convert_to_tensor(sample[0],dtype=np.float64) #float
smoking = tf.convert_to_tensor(sample[1]) #str
alcohol = tf.convert_to_tensor(sample[2]) #str
stroke = tf.convert_to_tensor(sample[3]) #str
physical = tf.convert_to_tensor(sample[4],dtype=np.int64) #int
mental = tf.convert_to_tensor(sample[5],dtype=np.int64) #int
walk = tf.convert_to_tensor(sample[6]) #str
sex = tf.convert_to_tensor(sample[7]) #str
age = tf.convert_to_tensor(sample[8]) #str
diabetic = tf.convert_to_tensor(sample[9]) #str
activity = tf.convert_to_tensor(sample[10]) #str
health = tf.convert_to_tensor(sample[11]) #str
sleep = tf.convert_to_tensor(sample[12],dtype=np.int64) #int
asthma = tf.convert_to_tensor(sample[13]) #str
kidney = tf.convert_to_tensor(sample[14]) #str
skinCancer = tf.convert_to_tensor(sample[15]) #str
X_dict = {'bmi': bmi, 'smoking': smoking,'alcoholDrinking': alcohol, 'stroke': stroke, 'physicalHealth': physical,
             'mentalHealth': mental, 'diffWalk': walk, 'sex': sex, 'ageGroup': age,'diabetic': diabetic, 'physicalActivity': activity, 
             'overallHealth': health, 'sleepHours': sleep, 'asthma': asthma,  'kidneyDisease': kidney, 'skinCancer': skinCancer}
X_ds = tf.data.Dataset.from_tensors((X_dict)).batch(1)
p_sample = model.predict(X_ds)
#%%
p_sample = tf.round(p_sample).numpy().flatten().reshape((1,1))
print(p_sample)
p_sample.shape
#%%
X = np.hstack((sample.to_numpy().reshape((1,) + X_train_resampled.iloc[13,:].shape), p_sample))
print(X)
print(z_test_pos[1])
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
z_test_pos_norm = z_test_pos.copy()
z_test_pos_norm = encode_strings(z_test_pos_norm)
print(X)
print(z_test_pos[1])
#%%
prototype_explainer = ProtodashExplainer()
(W,S,values) = prototype_explainer.explain(X_norm, z_test_pos_norm, m=5)

#%%
dataframe.pop('target')
#%%
dfs = pd.DataFrame.from_records(z_test_pos[S,0:-1])
dfs.columns = dataframe.columns
# %%
dfs['Weights'] = np.around(W, 5)/np.sum(np.around(W, 5))
# %%
dfs.transpose()
# %%
print(X.transpose())
# %%
