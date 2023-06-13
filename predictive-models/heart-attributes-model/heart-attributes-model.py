#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# %%
heart_csv_path = 'C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/data/heart.csv'
dataframe = pd.read_csv(heart_csv_path)
dataframe['corrected_target'] = np.where(dataframe['target']==0,1,0)
dataframe.pop('target')
dataframe.head()
#%%
train, val, test = np.split(dataframe.sample(frac=1), [int(0.5*len(dataframe)), int(0.9*len(dataframe))])
# train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
y_train = train.pop('corrected_target')
X_train = train
y_val = val.pop('corrected_target')
X_val = val
y_test = test.pop('corrected_target')
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
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
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
result = model.fit(X_train_NORM, y_train, 
          epochs=50,
          batch_size=32,
          validation_data=(X_val_NORM, y_val))

# %%
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
#%%
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()
model.evaluate(X_test_NORM, y_test) 
#%%
binary_predictions = tf.round(model.predict(X_test)).numpy().flatten()
print(classification_report(y_test, binary_predictions))
#%%
model.save("C:/Users/Rawan Alamily/Downloads/McSCert Co-op/explainable-ai-heart/predictive-models/heart-attributes-model/saved-model")

# %%
