# flood-prediction
A flood prediction system using machine learning.

## Importing
Please use `from [src.]model import FloodPrediction` to import. It is only necessary to use `src.model` from  outside the `src/` directory.

## Note
The model doesn't always train well on the first try. Allow for a few attempts or change the callback for early stopping to have a patience of 10, e.g.

```
model = FloodPrediction()

cb = [tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, patience=10, mode='max', restore_best_weights=True)]

model.tune_model(callbacks=cb)

model.fit_model(callbacks=cb)
```
