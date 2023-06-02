#%%
import tensorflow as tf
from tensorflow import keras
import shap
import numpy as np
import pandas as pd
#%%
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/saved-model")
#%%
X_train = pd.read_pickle("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/X_train_resampled.pkl")
X_test = pd.read_pickle('C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/X_test.pkl')
training_data = X_train.iloc[:10,:]

#%%
def f(X):
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
    X_dict = {'bmi': bmi, 'smoking': smoking,'alcoholDrinking': alcohol, 'stroke': stroke, 'physicalHealth': physical,
             'mentalHealth': mental, 'diffWalk': walk, 'sex': sex, 'ageGroup': age,'diabetic': diabetic, 'physicalActivity': activity, 
             'overallHealth': health, 'sleepHours': sleep, 'asthma': asthma,  'kidneyDisease': kidney, 'skinCancer': skinCancer}
    X_ds = tf.data.Dataset.from_tensor_slices((X_dict))
    X_ds = X_ds.batch(128)
    return model.predict(X_ds)
#%%
explainer = shap.KernelExplainer(f, data=training_data)
#%%
shap.initjs()
# get single input shap feature plot
shap_values = explainer.shap_values(X_test.iloc[1,:])
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[1,:])

#%%
# get for multiple input
shap_values = explainer.shap_values(X_test.iloc[:20,:])
shap.force_plot(explainer.expected_value, shap_values[0], X_test[:20])

# %%
# bar plot for multiple inputs
shap.summary_plot(shap_values, X_test, plot_type="bar")