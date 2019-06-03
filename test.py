import tensorflow as tf


posenet = tf.contrib.saved_model.load_keras_model('./saved_models/1559169309')



loss, mae, mse = posenet.evaluate(X_test, y_test)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

y_test_hat = posenet.predict(X_test)
residuals = y_test_hat - y_test
plt.hist(residuals['Yaw'], bins = 25)
plt.show()
