#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# %%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
train, val, test = np.split(dataframe.sample(frac=1), [int(0.6*len(dataframe)), int(0.9*len(dataframe))])
# train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test
print(len(X_train))
print(len(X_val))
print(len(X_test))
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_NORM = scaler.transform(X_train)
X_val_NORM = scaler.transform(X_val)
X_test_NORM = scaler.transform(X_test)

#%%
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=128, activation='relu'), 
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])
#%%
model.fit(X_train_NORM, y_train, 
          epochs=70,
          batch_size=32,
          validation_data=(X_val_NORM, y_val))
#%% 
model.evaluate(X_test_NORM, y_test) 
#%%
binary_predictions = tf.round(model.predict(X_test)).numpy().flatten()
print(classification_report(y_test, binary_predictions))
#%%
model.save("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-model")

# %%
