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
#plt.title("MinMax")
#plt.show()

# lag and forecast (TODO: rewrite)
LAG = 1
FORE = 1
cols, names = list(), list()
for i in range(LAG, 0, -1):
    cols.append(X.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(X.shape[1])] #NEXT: try fstring for j in X.columns
for i in range(0, FORE):
    cols.append(X.shift(-1))
if i == 0:
    names += [('var%d(t)' % (j+1)) for j in range(X.shape[1])]
else:
    names += [('var%d(t+%d)' % (j+1, i)) for j in range(X.shape[1])]
#X = pd.concat(cols,axis=1) #TODO: TypeError: '<' not supported between instances of 'str' and 'int'
#X.columns = names
X.dropna(inplace=True)

import numpy as np

# reshape data (TODO: rewrite)
seq = 20
dfX = []
dfY = []
for i in range(0,len(X) - seq):
    data = []
    for j in range(0,seq):
        d = []
        for col in X.columns:
            d.append(X[col][i +j])
        data.append(d)
    dfX.append(data)
    dfY.append(y[['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']].iloc[i + seq].values)
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
#model.add(tf.keras.layers.Dropout(0.2)) # drop data to control overfitting
model.add(tf.keras.layers.Dense(units=12, activation='sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

#print(X_train.shape[1])

# enable early stopping
cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# training
history = model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, callbacks=[cb])

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.ylabel("loss / accuracy")
plt.xlabel("Epoch")
plt.legend(['loss','accuracy'])
plt.title("Training")
plt.show()

# make prediction
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = pd.DataFrame(y_pred, columns=list(df[df.columns[-12:]]))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report#, confusion_matrix

# evaluate prediction
model.evaluate(X_test, y_test)

print("Actual:")
print(pd.DataFrame(y_test, columns=list(df[df.columns[-12:]])))
print("Predictions:")
print(y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))


