from keras.layers import SimpleRNN as SRNN
from sklearn.datasets import make_regression
import torch
from keras.layers import Dense, LSTM, Dropout, InputLayer
from keras.models import Sequential, Model
from collections import deque
from yahoo_fin import stock_info as yf  # Data preparation
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import time as tm
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as jinx
from subprocess import run
import os
from flask import Flask, render_template
# from keras.models import load_model

if np.__version__ >= '1.23.5':
    import tensorflow as tf


# If called from the jinx.py file do something, else do something else.
if __name__ == '__main__':
    os.system('clear')
    print('Jinx Testing...')
    sleep(1)
    os.system('clear')
    print('Jinx Testing Initiated')
    sleep(1)
    os.system('clear')
    print('Jinx Testing Start')
    sleep(1)
elif __name__ == 'jinx':
    os.system('clear')
    print('Jinx Server Initiated')
    sleep(1)
    os.system('clear')
    print('Jinx System Start')
    sleep(1)
    os.system('clear')
    print('Jinx System Active')
    sleep(1)
    os.system('clear')
else:
    exit(0)

# Stock ticker, GOOGL
# Get user input for the stock ticker
STOCKS = ('TSLA', 'AAPL', 'AMZN', 'MSFT', 'FB', 'GOOGL', 'GOOGL')
# Export name of this file as STOCKS_FILE to bash environment
os.environ['STOCKS_FILE'] = "__file__"
# Export STOCKS to bash environment
os.environ['STOCKS'] = ' '.join(STOCKS)
S = STOCKS[0]
if __name__ == '__main__':
    STOCK = input('Enter the stock ticker: ') or S
elif __name__ == 'jinx':
    STOCK = S

# Current date
date_now = tm.strftime('%Y-%m-%d')
date_3_years_back = (
    dt.date.today() - dt.timedelta(days=4*1104)).strftime('%Y-%m-%d')

# LOAD DATA
# from yahoo_fin
# for 1104 bars with interval = 1d (one day)
if __name__ == '__main__':
    # input_interval = input('Enter the interval (m, h, d, w, y): ') or 'd'
    input_interval = 'd'
elif __name__ == 'jinx':
    input_interval = 'd'
init_df = yf.get_data(STOCK, start_date=date_3_years_back,
                      end_date=date_now, interval='1' + input_interval)

# Copy the data to a new dataframe
jid = init_df.copy()

# Save the data as a candlestick chart
# Candlestick Park


jid_name = ['close', ['open', 'high', 'low', 'adjclose', 'ticker', 'volume']]
jvid00_name = ['volume', ['close', 'open', 'high', 'low', 'adjclose', 'ticker']]
jvid01_name = ['volume', ['open', 'high', 'low', 'close', 'adjclose', 'ticker']]
# remove columns which our neural network will not use
init_df = init_df.drop(jid_name[1], axis=1)
# create the column 'date' based on index column
init_df['date'] = init_df.index

# Scale data for ML engine
scaler = MinMaxScaler()
init_df[jid_name[0]] = scaler.fit_transform(
    np.expand_dims(init_df[jid_name[0]].values, axis=1))

# SETTINGS

# Window size or the sequence length, 7 (1 week)
N_STEPS = 7

# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3]


# Make direcotry in templates/ticker/interval+date.html
if not os.path.exists('templates/' + STOCK):
    os.makedirs('templates/' + STOCK, exist_ok=True)
init_df.to_html('templates/' + STOCK + str(dt.date.today()) +
                tm.strftime('%H:%M:%S.%f') + '.html')  # Save the data to html

if not os.path.exists('static/images'):
    os.makedirs('static/images', exist_ok=True)
# Save the plot
init_df.savefig('static/images/' + STOCK + str(dt.date.today()) + '_' + tm.strftime('%H:%M:%S.%f') + '.png')

# Let's preliminary see our data on the graphic
plt.style.use(style='ggplot')
plt.figure(figsize=(16, 10))
plt.plot(init_df[jid_name[0]][-2000:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for {STOCK}'])
plt.show()


def PrepareData(days):
    df = init_df.copy()
    df['future'] = df[jid_name[0]].shift(-days)
    last_sequence = np.array(df[[jid_name[0]]].tail(days))
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=N_STEPS)

    for entry, target in zip(df[[jid_name[0]] + ['date']].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == N_STEPS:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len([jid_name[0]])]
                         for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    # construct the X's and Y's
    X, Y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        Y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    return df, last_sequence, X, Y


def GetTrainedModel(x_train, y_train):
    # Define the input shape
    # input_shape = (x_train.shape[1], x_train.shape[2])
    model = Sequential()
    model.add(LSTM(60, return_sequences=True,
              input_shape=(N_STEPS, len([jid_name[0]]))))
    model.add(Dropout(0.3))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Dense(1))

    BATCH_SIZE = 20
    EPOCHS = 250

    model.compile(loss='mean_squared_error', optimizer='adam')

    # if run from index.py run as a thread
    if __name__ == '__main__':  # Running in jinx
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.submit(model.fit, x_train, y_train,
                            batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    else:  # Running in index
        model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                  epochs=EPOCHS, verbose=0)

    model.summary()

    return model


# GET PREDICTIONS
predictions = []

for step in LOOKUP_STEPS:
    df, last_sequence, x_train, y_train = PrepareData(step)
    x_train = x_train[:, :, :len([jid_name[0]])].astype(np.float32)

    model = GetTrainedModel(x_train, y_train)

    last_sequence = last_sequence[-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    predictions.append(round(float(predicted_price), 2))

# Execute model for the whole history range
copy_df = init_df.copy()
y_predicted = model.predict(x_train)
y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
first_seq = scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
last_seq = scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
copy_df['predicted_close'] = y_predicted_transformed

# Add predicted results to the table
date_now = dt.date.today()
date_tomorrow = dt.date.today() + dt.timedelta(days=1)
date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

copy_df.loc[date_now] = [predictions[0], f'{date_now}', 0]
copy_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0]
copy_df.loc[date_after_tomorrow] = [
    predictions[2], f'{date_after_tomorrow}', 0]

# Result chart
plt.style.use(style='ggplot')
plt.figure(figsize=(16, 10))
plt.plot(copy_df[jid_name[0]][-150:].head(147))
plt.plot(copy_df['predicted_close'][-150:].head(147),
         linewidth=1, linestyle='dashed')
plt.plot(copy_df[jid_name[0]][-150:].tail(4))
plt.annotate(f'{predictions[0]}', (date_now, predictions[0]),
             textcoords="offset points", xytext=(0, 10), ha='center')
plt.annotate(f'{predictions[1]}', (date_tomorrow, predictions[1]),
             textcoords="offset points", xytext=(0, 10), ha='center')
plt.annotate(f'{predictions[2]}', (date_after_tomorrow, predictions[2]),
             textcoords="offset points", xytext=(0, 10), ha='center')
plt.title(f'Predicted price for {STOCK}')
plt.table(cellText=[predictions], colLabels=['Today', 'Tomorrow', 'After Tomorrow'], rowLabels=[
          f'Predicted price for {STOCK}'], loc='bottom')
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f'Actual price for {STOCK}', f'Predicted price for {STOCK}',
           f'Predicted price for future 3 days'])
plt.show()

# Save the plot and the data to html
if not os.path.exists('templates/' + STOCK):
    os.makedirs('templates/' + STOCK, exist_ok=True)
copy_df.to_html('templates/' + STOCK + str(dt.date.today()) +
                tm.strftime('%H:%M:%S') + '.html')  # Save the data to html
plt.savefig('static/images/' + STOCK + str(dt.date.today()) + '_' +
            tm.strftime('%H:%M:%S') + '.png')

print("Copy: ", copy_df.tail(3))
print("Predictions: ", predictions[0:3])

# run(['sh', 'jinx.sh'])

# Import the data and preprocess it, then train the model and save it, also plot the data and save it as well as saving to html.


def train_model():
    # Import the data
    enddate = datetime.now()
    data = jinx.download('AAPL', start='2010-01-01', end=enddate)
    data = data['Close']
    data = data.to_frame()
    data = data.values
    data = data.astype('float32')
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    train_size = int(len(data) * 0.80)
    test_size = len(data) - train_size
    train, test = data[0:train_size, :], data[train_size:len(data), :]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # Train the model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])
    model.save('model.keras')
    # Predict the data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform([y_train])
    y_test = scaler.inverse_transform([y_test])
    # Plot the data
    look_back = 100
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2) +
                    1:len(data)-1, :] = test_predict
    plt.plot(scaler.inverse_transform(data))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig('static/images/' + STOCK + str(dt.date.today()) +
                '_' + tm.strftime('%H:%M:%S') + '.png')
    plt.close()
    # Save the data to html
    df = pd.DataFrame(data)
    df.to_html('templates/' + STOCK + str(dt.date.today()) +
               '_' + tm.strftime(' % H: % M: % S') + '.html')
    # Return the data
    return data


class Stock:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = yf.get_data(symbol)
        self.open = self.data['open']
        self.close = self.data['close']
        self.high = self.data['high']
        self.low = self.data['low']
        self.volume = self.data['volume']

    def get_open(self):
        return self.open

    def get_close(self):
        return self.close

    def get_high(self):
        return self.high

    def get_low(self):
        return self.low

    def get_volume(self):
        return self.volume

    def get_data(self):
        return self.data

    def get_symbol(self):
        return self.symbol

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction

# class to store the stock data and the prediction data to sqlite


class StockData:
    def __init__(self, symbol, open, close, high, low, volume, prediction):
        self.symbol = symbol
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.prediction = prediction

    def get_symbol(self):
        return self.symbol

    def get_open(self):
        return self.open

    def get_close(self):
        return self.close

    def get_high(self):
        return self.high

    def get_low(self):
        return self.low

    def get_volume(self):
        return self.volume

    def get_prediction(self):
        return self.prediction

# Download historical data for desired ticker symbol
# data = yf.download('AAPL', start='2020-01-01', end='2022-12-31')


def train():
    # Generate some example data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize weights
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # Set hyperparameters
    alpha = 1.0
    lambda_ = 1.0

    # Training loop
    for _ in range(1000):
        # Compute predictions
        y_pred = w * X + b

        # Compute loss
        loss = torch.mean((y - y_pred)**2) + alpha * w**2 + lambda_ * b**2

        # Backpropagation
        loss.backward()

        # Update weights
        with torch.no_grad():
            w -= 0.01 * w.grad
            b -= 0.01 * b.grad

        # Reset gradients
        w.grad.zero_()
        b.grad.zero_()

    print(w, b)

# Jinx Predictive Software


class StockStorage:
    def __init__(self, symbol, open, close, high, low, volume, prediction):
        self.symbol = symbol
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.prediction = prediction

    def get_symbol(self):
        return self.symbol

    def get_open(self):
        return self.open

    def get_close(self):
        return self.close

    def get_high(self):
        return self.high

    def get_low(self):
        return self.low

    def get_volume(self):
        return self.volume

    def get_prediction(self):
        return self.prediction


class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        self.__ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(
            self.__ticker, start=self.start_date, end=self.end_date)
        self.target = self.data['Close'].values
        self.scaler = MinMaxScaler()
        self.target = self.scaler.fit_transform(self.target.reshape(-1, 1))
        self.X = np.array([self.target[i-60:i, 0]
                          for i in range(60, len(self.target))])
        self.y = np.array([self.target[i, 0]
                          for i in range(60, len(self.target))])
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)


class LSTMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(50, return_sequences=False)
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)

    def predict(self, model_type="rnn"):
        if model_type.lower() == "rnn":
            model = self.SimpleRNNModel()
        elif model_type.lower() == "lstm":
            model = self.LSTMModel()
        else:
            raise ValueError("Invalid model_type. Expected 'rnn' or 'lstm'")

        return model(self.X[:1])
