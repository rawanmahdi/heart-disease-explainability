#%%
import tensorflow as tf
from tensorflow import keras
import shap
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
#%%
##NEWER MODEL 
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/life-heart.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
#%%
diff = list(np.where(dataframe['diffWalk']=='Yes'))
nodiff = np.where(dataframe['diffWalk'] == 'No')
heart = list(np.where(dataframe['target'] == 1))
#%%
heart_diff = [i for i in diff if i in heart]

#%%
d = np.where(dataframe['diabetic']=='Yes',1,0)
s = np.where(dataframe['smoking']=='Yes',1,0)
a =  np.where(dataframe['physicalActivity']=='No',1,0)
dataframe['BDSA'] = dataframe['bmi']*(d+s+a)
print('bmi:')
print(dataframe['bmi'].iloc[0:20])
print('bdsa')
print(dataframe['BDSA'].iloc[0:20])
dataframe = dataframe.drop(columns=['alcoholDrinking','kidneyDisease', 'skinCancer', 'sleepHours'])
neg, pos = np.bincount(dataframe['target'])
df = dataframe.copy()
# designate for fitting rus
y = df.pop('target')
X = df
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
# observe class imbalance
neg, pos = np.bincount(y_train)
print("No.negative samples before undersampling",neg)
print("No.positive samples before undersampling",pos)
# resample via undersampling majority class - this is favoured over oversampling as the dataset is very large
rus = RandomUnderSampler(random_state=0)
rus.fit(X,y)
# only resample training dataset
X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)
neg0, pos0 = np.bincount(y_train_resampled)
print("No.negative samples after undersampling",neg0)
print("No.positive samples after undersampling",pos0)

#%%
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/saved-altred-model")
#%%
training_data = X_train.iloc[:10,:]

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
explainer = shap.KernelExplainer(f, data=training_data)
#%%
# get single input shap feature plot
shap_values = explainer.shap_values(X_test.iloc[1,:])
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[1,:])
#%%
# get for multiple input
shap_values = explainer.shap_values(X_test.iloc[:200,:]);
print("done finding shap values")
#%%
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[:200])

# %%
# bar plot for multiple inputs
shap.summary_plot(shap_values, X_test, plot_type="bar")
# %%
