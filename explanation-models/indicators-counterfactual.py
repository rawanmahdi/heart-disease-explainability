#%%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dice_ml
from imblearn.under_sampling import RandomUnderSampler

#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/life-heart.csv'
dataframe = pd.read_csv(heart_csv_path)
print(dataframe.describe())
print(dataframe.shape)
dataframe['target'] = np.where(dataframe['heartDisease']=='Yes', 1, 0)
dataframe = dataframe.drop(columns=['heartDisease'])
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
neg, pos = np.bincount(dataframe['target'])
#%%
df = dataframe.copy()
# designate for fitting rus
y = df.pop('target')
X = df
#%%
ageCat =[]
for item in dataframe['ageGroup']:
    if item not in ageCat:
        ageCat.append(item)

print(len(ageCat))
#%%
train, val, test = np.split(dataframe.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test

# %%
# resample via undersampling majority class - this is favoured over oversampling as the dataset is very large
rus = RandomUnderSampler(random_state=0)
rus.fit(X,y)
# only resample training dataset
X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)
neg0, pos0 = np.bincount(y_train_resampled)
print("No.negative samples after undersampling",neg0)
print("No.positive samples after undersampling",pos0)
#%%
# train = X_train_resampled
# train['target'] = y_train_resampled
def df_to_dataset(features, labels, batch_size=512):
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).cache()
    shuffled_tf_dataset = tf_dataset.shuffle(buffer_size=len(df)) # shuffling values 
    return shuffled_tf_dataset.batch(batch_size).prefetch(2)# returning 32 samples per batch
train = df_to_dataset(X_train_resampled,y_train_resampled)
#%%
# build dice data object
dice_data = dice_ml.Data(dataframe=train, 
                         continuous_features=['bmi','physicalHealth','mentalHealth','sleepHours'],
                         outcome_name='target')
#%%
dice_data = dice_ml.Data(features={'bmi': [12,94.8], 
                        'smoking': ['Yes','No'],
                        'alcoholDrinking': ['Yes','No'], 
                        'stroke': ['Yes','No'], 
                        'physicalHealth': [0,30],
                        'mentalHealth': [0,30],
                        'diffWalk': ['Yes','No'],
                        'sex': ['Female','Male'], 
                        'ageGroup': ['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29'],
                        'diabetic': ['Yes','No', 'No, borderline diabetes'],
                        'physicalActivity': ['Yes','No'], 
                        'overallHealth': ['Excellent','Very good', 'Good', 'Fair', 'Poor'],
                        'sleepHours': [0,24], 
                        'asthma': ['Yes','No'],  
                        'kidneyDisease': ['Yes','No'], 
                        'skinCancer': ['Yes','No']},
                        outcome_name='target',
                        continuous_features=['bmi','physicalHealth','mentalHealth','sleepHours'])
#%%
# build dice model object
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/saved-model")

dice_model = dice_ml.Model(model=model,
                           backend='TF2')
# %%
explainer = dice_ml.Dice(dice_data,
                         dice_model,
                         method='random')
# %%
sample = X_test.iloc[20, :]

# %%
query_instance = X_train.iloc[0:1]
print(query_instance)
#%%
counterfactual1 = explainer.generate_counterfactuals(query_instance,
                                                    total_CFs=3,
                                                    desired_class='opposite')

counterfactual1.visualize_as_dataframe(show_only_changes=True)

# %%
# apply resdtrictions on feature variation
features_to_vary = ['trestbps','chol','thalach','oldpeak','ca']
permitted_range = {'trestbps':[80,200]}

counterfactual2 = explainer.generate_counterfactuals(query_instance,
                                                    total_CFs=3,
                                                    desired_class='opposite',
                                                    features_to_vary=features_to_vary,
                                                    permitted_range=permitted_range)

counterfactual2.visualize_as_dataframe(show_only_changes=True)

