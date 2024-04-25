import pandas as pd

# read data
df_rain = pd.read_csv("../data/chennai-monthly-rains.csv")
df_flood = pd.read_csv("../data/chennai-monthly-manual-flood.csv")

# clean data
df_rain = df_rain.drop(columns=['Total'])
df_flood = df_flood[(df_flood.year <= 2021)]

# merge datasets
#   TODO: remove second 'year' column added by merging
df = pd.merge(df_rain,df_flood, left_on='Year', right_on='year')

# check shape
print("Original shape:", df.shape)

import matplotlib.pyplot as plt

# visualise data
#df_rain.plot(x="Year")
#df_flood.plot(x="year")
#plt.show()

# data-target split
X = df[df.columns[:-12]].drop(columns=["Year","year"]) #df_rain.drop(columns="Year").values
y = df[df.columns[-12:]] #df_flood.values

# check shape after sampling target data
print("X:",X.shape)
print("y:",y.shape)

### TODO: should data be balanced here? How can mostly false (0) be balanced?

from sklearn.preprocessing import MinMaxScaler

# normalise data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) #scaler.fit_transform(X_train.drop(columns="Year"))
X = pd.DataFrame(X_scaled, columns=list(X.columns))

#plt.plot(X)
#plt.show()

import numpy as np

# reshape data
seq = 20
dfX = []
dfY = []
for i in range(0,len(df) - seq):
    data = []
    for j in range(0,seq):
        d = []
        for col in df.columns:
            d.append(df[col][i +j])
        data.append(d)
    dfX.append(data)
    dfY.append(df[['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']].iloc[i + seq].values)
X, y = np.array(dfX), np.array(dfY)

# check shape after reshaping
print("X:",X.shape)
print("y:",y.shape)

from sklearn.model_selection import train_test_split

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.1, random_state=42)

# check shape after splitting
print("X_train:",X_train.shape,"X_test:",X_test.shape)
print("y_train:",y_train.shape,"y_test:",y_test.shape)
print("X_val:",X_val.shape,"y_val:",y_val.shape)


import tensorflow as tf

# force GPU
#with tf.device('/GPU:0'):

# LSTM
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dropout(0.2)) # control overfitting
model.add(tf.keras.layers.Dense(units=12, activation='sigmoid'))
model.compile(loss='MeanSquaredError', optimizer='Adam')
model.summary()

#print(X_train.shape[1])

# enable early stopping
cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# training
history = model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, callbacks=[cb])

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("Epoch")
plt.show()

# make prediction
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=list(df[df.columns[-12:]]))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# evaluate prediction
print("Actual:")
print(pd.DataFrame(y_test, columns=list(df[df.columns[-12:]])))
print("Predictions:")
print(y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
