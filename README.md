# flood-prediction
A flood prediction system using machine learning.

## Importing
Please use `from [src.]model import FloodPrediction` to import. It is only necessary to use `src.model` from  outside the `src/` directory.

## Requirements
```
imbalanced_learn==0.12.2
keras_tuner==1.4.7
numpy==1.21.5
pandas==1.5.3
scikit_learn==1.2.2
tensorflow==2.16.1
```
Try:
```
pip install imbalanced_learn keras_tuner numpy pandas scikit_learn tensorflow
```
## Usage
The current version will likely throw an error on Windows due to the default filepath.

Therefore, it may be necessary to specify the path when calling `FloodPrediction()` like so:

```
model = FloodPrediction(path="..\data\")
```

Tuning the model will also likely need an explicit directory like so:

```
model.tune_model(directory="..\build\")
```

As will saving and loading:

```
model.save(filepath="..\build\model.keras")
model.load_save(filepath="..\build\model.keras")
```
## Note
The model doesn't always train well on the first try. Allow for a few attempts or change the callback for early stopping to have a patience of 10, e.g.

```
model = FloodPrediction()

cb = [tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, patience=10, mode='max', restore_best_weights=True)]

model.tune_model(callbacks=cb)

model.fit_model(callbacks=cb)
```
