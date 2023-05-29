#%%
import pandas as pd
import numpy as np
#%%
path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/personal-indicators-model/data/life-heart.csv'
df = pd.read_csv(path)
df['target'] = np.where(df['heartDisease']=='Yes', 1, 0)
df = df.drop(columns=['heartDisease'])
df['explanation'] = np.where(df['heartDisease']=='Yes' & df['bmi']>35.0, 
                             'BMI levels > 35.0 increase risk of heart disease', 
                             'Healthy')


