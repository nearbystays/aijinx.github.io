import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch data
data = yf.download('TSLA', start='2020-01-01', end='2022-01-01')

# Preprocess data
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - data['Date'].min()) / np.timedelta64(1, 'D')

X = data['Date'].values.reshape(-1, 1)
y = data['Close'].values

# Convert to TensorFlow tensors
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Define the model
model = tfp.glm.BayesianGLM(
    model_matrix=X_tensor,
    response=y_tensor,
    model=tfp.glm.Normal()
)

# Fit the model
coeffs, linear_response, is_converged, num_iter = model.fit()

# Print the coefficients
print("Coefficients:", coeffs.numpy())

# Predict prices
y_pred = linear_response.numpy()

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Actual prices')
plt.plot(X, y_pred, label='Predicted prices')
plt.legend()
plt.show()
