#%%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import dice_ml
import numpy as np

#%%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
dataframe['corrected_target'] = np.where(dataframe['target']==0,1,0)
dataframe.pop('target')
dataframe.head()
#%%
train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
train_dice = train.copy()
y_train = train.pop('corrected_target')
X_train = train
y_test = test.pop('corrected_target')
X_test = test

#%%
# build dice data object
dice_data = dice_ml.Data(dataframe=train_dice, 
                         continuous_features=['age','trestbps','chol','thalach','oldpeak','ca'],
                         outcome_name='corrected_target')
# %%
# build dice model object
model = keras.models.load_model("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-model")

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
features_to_vary = ['trestbps','chol', 'fbs', 'cp', 'restecg', 'oldpeak', 'slope', 'thal']
# permitted_range = {'trestbps':[80,200]}

counterfactual2 = explainer.generate_counterfactuals(query_instance,
                                                    total_CFs=3,
                                                    desired_class='opposite',
                                                    features_to_vary=features_to_vary,
                                                    # permitted_range=permitted_range
                                                    )

counterfactual2.visualize_as_dataframe(show_only_changes=True)

# %%
