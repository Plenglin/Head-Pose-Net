import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib.pyplot as plt


import model


EPOCHS = 1000

inputs = pd.read_csv('ml_inputs.csv', index_col=0)
outputs = pd.read_csv('ml_outputs.csv', index_col=0)
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, shuffle=True)

model = model.create_model()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
model.fit(
    X_train, y_train,
    epochs=EPOCHS, 
    validation_split=0.2,
    batch_size=32, 
    callbacks=[early_stop])

loss, mae, mse = model.evaluate(X_test, y_test)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

y_test_hat = model.predict(X_test)
residuals = y_test_hat - y_test
plt.hist(residuals['Yaw'], bins = 25)
plt.show()
