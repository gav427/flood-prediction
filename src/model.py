import pandas as pd

# read data
df_rain = pd.read_csv("../data/chennai-monthly-rains.csv")
df_flood = pd.read_csv("../data/chennai-monthly-manual-flood.csv")

# clean data
df_flood = df_flood[(df_flood.year <= 2021)]

# merge datasets
#   TODO: remove second 'year' column added by merging
df = pd.merge(df_rain,df_flood, left_on='Year', right_on='year')

# concat columns
df = pd.concat([df, df[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]].T.stack().reset_index(name='months_rain')['months_rain']], axis=1)
df = pd.concat([df, df[["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]].T.stack().reset_index(name='months_flood')['months_flood']], axis=1)

# drop unnecessary columns
df.drop(inplace=True, columns=["Year","Total","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
df.drop(inplace=True, columns=["year","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])

# stats
print(df[['months_rain','months_flood']].describe())

# check shape
print("Original shape:", df.shape)

import matplotlib.pyplot as plt

# visualise data
#df_rain.plot(x="Year")
#df_flood.plot(x="year")
#plt.show()

import numpy as np

# check data balance
# Adapted from: “Classification on imbalanced data | TensorFlow Core,” TensorFlow. https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
neg, pos = np.bincount(df['months_flood'])
t = neg + pos
print(f"Total: %d; positive: %d (%.2f%% of total)" % (t,pos,(100*pos/t)))

# TODO: convert data to log-space?

# data-target split
#   TODO: should this be done after data splitting?
X = df[df.columns[:-1]] #.drop(columns=["Year","year"]) #df_rain.drop(columns="Year").values
y = df[df.columns[-1:]] #df_flood.values

# check shape after sampling target data
print("X:",X.shape)
print("y:",y.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# z-score
std_scaler = StandardScaler()
X_zerod = std_scaler.fit_transform(X)
X = pd.DataFrame(X_zerod, columns=list(X.columns))

# normalise data
minmax_scaler = MinMaxScaler()
X_scaled = minmax_scaler.fit_transform(X) #scaler.fit_transform(X_train.drop(columns="Year"))
X = pd.DataFrame(X_scaled, columns=list(X.columns))

#plt.plot(X)
#plt.title("MinMax")
#plt.show()

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# over- (SMOTE) and under-sampling
oversample = SMOTE(sampling_strategy=0.1, k_neighbors=2)
understample = RandomUnderSampler()

X, y = oversample.fit_resample(X, y); print("Y",y)
X, y = understample.fit_resample(X, y)

# lag and forecast
# Adapted from: J. Brownlee, “How to Convert a Time Series to a Supervised Learning Problem in Python,” Machine Learning Mastery, May 07, 2017. https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
LAG = 1
FORE = 1
cols, names = list(), list()
for i in range(LAG, 0, -1):
    cols.append(X.shift(i))
    names += [(f'var%d(t-%d)' % (j+1, i)) for j in range(X.shape[1])] #NEXT: try fstring for j in X.columns
for i in range(0, FORE):
    cols.append(X.shift(-1))
if i == 0:
    names += [(f'var%d(t)' % (j+1)) for j in range(X.shape[1])]
else:
    names += [(f'var%d(t+%d)' % (j+1, i)) for j in range(X.shape[1])]
X = pd.concat(cols,axis=1)
X.columns = names
X.dropna(inplace=True)

# reshape data
# Adapted from: S. S. Bhakta, “Multivariate Time Series Forecasting with LSTMs in Keras,” GeeksforGeeks, Feb. 17, 2024. https://www.geeksforgeeks.org/multivariate-time-series-forecasting-with-lstms-in-keras/ (accessed May 02, 2024).
seq = 20
dfX = []
dfY = []
for i in range(0,len(X) - seq):
    data = [[X[col].iloc[i+j] for col in X.columns] for j in range(0,seq)]
    dfX.append(data)
    dfY.append(y[['months_flood']].iloc[i + seq].values)
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
import keras_tuner as kt

def build_model(hp):

    # hyper parameter tuning
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_units_2 = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # force GPU
    #with tf.device('/GPU:0'):

    # define metrics
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.BinaryCrossentropy(name='log loss'),
        tf.keras.metrics.TruePositives(name='TP'),
        tf.keras.metrics.TrueNegatives(name='TN'),
        tf.keras.metrics.FalsePositives(name='FP'),
        tf.keras.metrics.FalseNegatives(name='FN'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='prc', curve='PR')
    ]

    # LSTM
    model = tf.keras.Sequential()
    model.add(tf.keras.Input((X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=hp_units_2))
    model.add(tf.keras.layers.Dropout(0.2)) # drop data to control overfitting
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(np.log([pos/neg]))))
    model.compile(loss='BinaryCrossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=METRICS)
    model.summary()

    #print(X_train.shape[1])

    return model

# enable early stopping
cb = tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, patience=3, mode='max', restore_best_weights=True)

# hyperparameter search
tuner = kt.Hyperband(build_model, objective=kt.Objective('prc', direction='max'), max_epochs=10, factor=3)
tuner.search(X_train, y_train, epochs=50, callbacks=[cb])
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hp.get('units'), best_hp.get('learning_rate'))

# use best model
model = tuner.hypermodel.build(best_hp)

# create class weights
# Adapted from: “Classification on imbalanced data: Tensorflow Core,” TensorFlow, https://www.tensorflow.org/tutorials/structured_data/imbalanced_data (accessed May 2, 2024). 
#weight_neg = (1 / neg) * (t / 2.0)
#weight_pos = (1 / pos) * (t / 2.0)

#weights = {0: weight_neg, 1: weight_pos}; print("Weights:", weights)

# training
history = model.fit(X_train, y_train, epochs=20, batch_size=1, validation_data=(X_val, y_val), verbose=1, callbacks=[cb]) #, class_weight=weights)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.ylabel("loss / accuracy")
plt.xlabel("Epoch")
plt.legend(['loss','accuracy'])
plt.title("Training")
plt.show()

# make prediction
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.9, 1, 0) # magic number = 0.026; with weights = ~0.5325
y_pred = pd.DataFrame(y_pred, columns=list(df[df.columns[-1:]]))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

# evaluate prediction
model.evaluate(X_test, y_test)

print("Actual:")
print(pd.DataFrame(y_test, columns=list(df[df.columns[-1:]])))
print("Predictions:")
print(y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

