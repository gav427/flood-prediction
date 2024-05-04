import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix


class FloodPrediction:
    '''This class creates an LSTM model that predicts flood events based on \
    monthly rainfall data'''

    def __init__(self, path="../data/", data_file="chennai-monthly-rains.csv", target_file="chennai-monthly-manual-flood.csv"):
        self._df = df
        self._X = pd.DataFrame()
        self._y = pd.DataFrame()
        self._X_train
        self._X_test
        self._X_val
        self._y_train
        self._y_test
        self._y_val
        self.lag
        self.fore
        self.sequence

        # read files and format data
        self.load_data(data_file, target_file)
        
        # data-target split
        self.make_data_target()
        
        # z-score
        self.apply_standard_scaler()
            
    def load_data(data, target,
                  data_index='Year',
                  target_index='year',
                  collapse_data_columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                  collapse_target_columns=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                  data_column_name='months_rain',
                  target_column_name='months_flood',
                  drop_data_columns=["Year", "Total"],
                  drop_target_columns=["year"]):
        '''This method loads training data and supervised learning targets.'''
    
        # read data
        df_rain = pd.read_csv(path + data)
        df_flood = pd.read_csv(path + target)

        # clean data
        if df_rain[data_index].max() < df_flood[target_index].max():
            df_flood = df_flood[(df_flood[target_index] <= df_rain[data_index].max())]

        if df_rain[data_index].max() > df_flood[target_index].max():
            df_rain = df_rain[(df_rain[data_index] <= df_flood[target_index].max())]

        # merge datasets
        #   TODO: remove second 'year' column added by merging
        df = pd.merge(df_rain, df_flood, left_on=data_index, right_on=target_index)

        # concat columns
        df = pd.concat([df, df[collapse_data_columns].T.stack().reset_index(name=data_column_name)[data_column_name]], axis=1)
        df = pd.concat([df, df[collapse_target_columns].T.stack().reset_index(name=data_column_name)[target_column_name]], axis=1)

        # drop unnecessary columns
        df.drop(inplace=True, columns=drop_data_columns + collapse_data_columns)
        df.drop(inplace=True, columns=drop_target_columns + collapse_target_columns)

    def make_data_target(self):
        '''This method splits the dataframe into data (X) and supervised learning targets (y).'''
        X = df[df.columns[:-1]]
        y = df[df.columns[-1:]]

    def apply_standard_scaler(self):
        '''This method applies the standard scaler to the data.'''

        std_scaler = StandardScaler()
        X_zerod = std_scaler.fit_transform(X)
        X = pd.DataFrame(X_zerod, columns=list(X.columns))

    def get_dataframe(self):
        '''Accessor returns the raw pandas dataframe used for training the model.'''
        return _df

    def get_data(self):
        '''Accessor returns the data used to train the model.'''
        return _X

    def get_target(self):
        '''Accessor returns the supervised learning targets/labels used to train the model.'''
        return _y

    def get_data_train_split(self):
        '''Accessor returns the data training split after data splitting used for training the model.'''
        return _X_train

    def get_target_train_split(self):
        '''Accessor returns the target training split after data splitting used for training the model.'''
        return _y_train

    def get_data_test_split(self):
        '''Accessor returns the data test split after data splitting used for training the model.'''
        return _X_test

    def get_target_test_split(self):
        '''Accessor returns the target test split after data splitting used for training the model.'''
        return _y_test

    def get_data_validation_split(self):
        '''Accessor returns the data validation split after data splitting used for validating the trained model.'''
        return _X_val

    def get_target_validation_split(self):
        '''Accessor returns the target validation split after data splitting used for validating the trained model.'''
        return _y_val


if __name__ == "__main__":

    # create a model for testing
    new_model = FloodPrediction()

    # stats
    print(df[['months_rain', 'months_flood']].describe())

    # check shape
    print("Original shape:", df.shape)

    # visualise data
    #df_rain.plot(x="Year")
    #df_flood.plot(x="year")
    #plt.show()

    # check data balance
    # Adapted from: “Classification on imbalanced data | TensorFlow Core,” TensorFlow. https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    neg, pos = np.bincount(df['months_flood'])
    t = neg + pos
    print(f"Total: %d; positive: %d (%.2f%% of total)" % (t, pos, (100*pos/t)))

    # TODO: convert data to log-space?

    # check shape after sampling target data
    print("X:", X.shape)
    print("y:", y.shape)

    # normalise data
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=list(X.columns))

    #plt.plot(X)
    #plt.title("MinMax")
    #plt.show()

    # over- (SMOTE) and under-sampling
    oversample = SMOTE(sampling_strategy=0.1, k_neighbors=2)
    understample = RandomUnderSampler()

    X, y = oversample.fit_resample(X, y); print("Y", y)
    X, y = understample.fit_resample(X, y)

    # lag and forecast
    # Adapted from: J. Brownlee, “How to Convert a Time Series to a Supervised Learning Problem in Python,” Machine Learning Mastery, May 07, 2017. https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    LAG = 1
    FORE = 1
    cols, names = list(), list()
    for i in range(LAG, 0, -1):
        cols.append(X.shift(i))
        names += [(f'var%d(t-%d)' % (j+1, i)) for j in range(X.shape[1])]
    for i in range(0, FORE):
        cols.append(X.shift(-1))
    if i == 0:
        names += [(f'var%d(t)' % (j+1)) for j in range(X.shape[1])]
    else:
        names += [(f'var%d(t+%d)' % (j+1, i)) for j in range(X.shape[1])]
    X = pd.concat(cols, axis=1)
    X.columns = names
    X.dropna(inplace=True)

    # reshape data
    # Adapted from: S. S. Bhakta, “Multivariate Time Series Forecasting with LSTMs in Keras,” GeeksforGeeks, Feb. 17, 2024. https://www.geeksforgeeks.org/multivariate-time-series-forecasting-with-lstms-in-keras/ (accessed May 02, 2024).
    seq = 20
    dfX = []
    dfY = []
    for i in range(0, len(X) - seq):
        data = [[X[col].iloc[i+j] for col in X.columns] for j in range(0, seq)]
        dfX.append(data)
        dfY.append(y[['months_flood']].iloc[i + seq].values)
    X, y = np.array(dfX), np.array(dfY)

    # check shape after reshaping
    print("X:", X.shape)
    print("y:", y.shape)

    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # check shape after splitting
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)


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
        
        # drop data to control overfitting
        model.add(tf.keras.layers.Dropout(0.2))
        
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
    plt.legend(['loss', 'accuracy'])
    plt.title("Training")
    plt.show()

    # make prediction
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.9, 1, 0) # magic number = 0.026; with weights = ~0.5325
    y_pred = pd.DataFrame(y_pred, columns=list(df[df.columns[-1:]]))

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
