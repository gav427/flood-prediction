import pandas as pd

# read data
df_rain = pd.read_csv("../data/chennai-monthly-rains.csv")
df_flood = pd.read_csv("../data/chennai-monthly-manual-flood.csv")

# clean data
df_rain = df_rain.drop(columns=['Total'])
df_flood = df_flood[(df_flood.year <= 2021)]

import matplotlib.pyplot as plt

# visualise data
df_rain.plot(x="Year")
df_flood.plot(x="year")
plt.show()

# convert to arrays
X = df_rain.drop(columns="Year").values
y = df_flood.values

### TODO: should data be balanced here? How can mostly false (0) be balanced?

from sklearn.preprocessing import MinMaxScaler

# normalise data
#   TODO: replace sklearn with tf
#         retain year coloumn, months if possible
scaler = MinMaxScaler()
X = scaler.fit_transform(X) #scaler.fit_transform(X_train.drop(columns="Year"))
#X_train = pd.DataFrame(X_train_normalised)

plt.plot(X)
plt.show()

from sklearn.model_selection import train_test_split

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.1, random_state=42)

# TODO: create sequences

'''

import tensorflow as tf

# force GPU
#with tf.device('/GPU:0'):

# LSTM
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=128, input_shape=(X_train.shape[0], X_train.shape[1])))
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.Dense(units=1))

model.compile(loss='mean_square_error')

model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("Epoch")
plt.show()

'''
