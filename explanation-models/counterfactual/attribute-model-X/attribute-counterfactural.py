#%%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import dice_ml
import numpy as np

#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart.csv'
good_df = pd.read_csv(heart_csv_path)
good_df['Sex'] = np.where(good_df['Sex'] == "M",1,0)
good_df['ExerciseAngina'] = np.where(good_df['ExerciseAngina'] == "Y",1,0)
def cp(x):
    if x=='TA':
        y=1
    elif x=='ATA':
        y=0
    elif x=='NAP':
        y=2
    else:
        y=3
    return y

def restECG(x):
    if x=='Normal':
        y=1
    elif x=='ST':
        y=0
    else:
        y=2
    return y

def slope(x):
    if x=='Up':
        y=1
    elif x=='Flat':
        y=0
    else:
        y=2
    return y
#%%
for i in range(good_df.iloc[:, 2].shape[0]):
    good_df.iloc[i,2] = cp(good_df.iloc[i,2])
for i in range(good_df.iloc[:, 6].shape[0]):
    good_df.iloc[i,6] = restECG(good_df.iloc[i,6])
for i in range(good_df.iloc[:, 10].shape[0]):
    good_df.iloc[i,10] = slope(good_df.iloc[i,10])
#%%
train, test = train_test_split(good_df, test_size=0.2, random_state=42)
train_dice = train.copy()
y_train = train.pop('HeartDisease')
X_train = train
y_test = test.pop('HeartDisease')
X_test = test

#%%
# build dice data object
dice_data = dice_ml.Data(dataframe=train_dice, 
                         continuous_features=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak'],
                         outcome_name='HeartDisease')
# %%
# build dice model object
good_model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-model")

dice_model = dice_ml.Model(model=good_model,
                           backend='TF2')
# %%
explainer = dice_ml.Dice(dice_data,
                         dice_model,
                         method='random')
# %%
query_instance = X_train.iloc[18:19]
print(y_train[18])
print(query_instance)
#%%
counterfactual1 = explainer.generate_counterfactuals(query_instance,
                                                    total_CFs=3,
                                                    desired_class='opposite')

counterfactual1.visualize_as_dataframe(show_only_changes=True)

# %%
# apply resdtrictions on feature variation
features_to_vary = ['RestingBP','Cholesterol','MaxHR',"RestingECG"]
permitted_range = {'RestingBP':[80,200], 'Cholesterol':[50,200], 'MaxHR':[130,200]}

counterfactual2 = explainer.generate_counterfactuals(query_instance,
                                                    total_CFs=3,
                                                    desired_class='opposite',
                                                    features_to_vary=features_to_vary,
                                                    permitted_range=permitted_range
                                                    )

counterfactual2.visualize_as_dataframe(show_only_changes=True)

# %%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart_messed_up.csv'
bad_df = pd.read_csv(heart_csv_path)
bad_model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-bad-model")
train, val, test = np.split(bad_df.sample(frac=1), [int(0.5*len(bad_df)), int(0.9*len(bad_df))])
# train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
train_dice = train.copy()
y_train = train.pop('target')
X_train = train
y_test = test.pop('target')
X_test = test
#%%
# build dice data object
dice_data = dice_ml.Data(dataframe=train_dice, 
                         continuous_features=['age','trestbps','chol','thalach','oldpeak'],
                         outcome_name='target')
# %%
# build dice model object
dice_model = dice_ml.Model(model=bad_model,
                           backend='TF2')
# %%
explainer = dice_ml.Dice(dice_data,
                         dice_model,
                         method='random')
# %%
query_instance = X_train.iloc[7:8]
print(y_train[7])
print(query_instance)
#%%
counterfactual1 = explainer.generate_counterfactuals(query_instance,
                                                    total_CFs=3,
                                                    desired_class='opposite')

counterfactual1.visualize_as_dataframe(show_only_changes=True)

# %%
