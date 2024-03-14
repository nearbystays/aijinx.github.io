# from jinx import MinMaxScaler, pd, plt, np, tf, os, datetime, warnings, jid
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from datetime import datetime
import warnings
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from time import sleep

warnings.filterwarnings("ignore")
companies = ['AAPL', 'GOOGL', 'GOOG', 'AMZN', 'MSFT', 'META', 'TSLA']

# convert the data to csv
data = yf.download(companies[-1], start="2010-01-01",
                   end=datetime.now(), interval="1d")

print(data.shape)
print(data.sample(7))
# sleep(60)
# Print data keys
print(data.keys())
# sleep(60)
data['Date'] = data.index

# date vs open
# date vs close

plt.figure(figsize=(15, 8))

for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['Date'], c['Close'], c="r", label="Close", marker="+")
    plt.plot(c['Date'], c['Open'], c="g", label="Open", marker="^")
    plt.title(company)
    plt.legend()
    plt.tight_layout()

plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['Date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.tight_layout()

apple = data[data['Name'] == 'AAPL']
prediction_range = apple.loc[(apple['Date'] > datetime(
    2013, 1, 1)) & (apple['Date'] < datetime(2018, 1, 1))]
plt.plot(apple['Date'], apple['Close'])
plt.xlabel("Date")
plt.ylabel("Close")

plt.title("Apple Stock Prices")
plt.show()
# Save plot to static/images/company/date.png
plt.savefig(f"static/images/{company}/{datetime.now()}.png")
close_data = apple.filter(['Close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)


scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)


train_data = scaled_data[0:int(training), :]
# prepare feature and labels

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=10)

test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

train = apple[:training]
test = apple[training:]
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['Date'], train['Close'])
plt.plot(test['Date'], test[['Close', 'Predictions']])
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()
plt.savefig(f"static/images/{company}/{datetime.now()}.png")
