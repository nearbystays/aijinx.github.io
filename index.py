import yfinance as yf
import sqlite3
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tabulate import tabulate
import argparse
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
import subprocess
from time import sleep
import asyncio
import multiprocessing

# List of tickers
# Fetch the data
# for ticker in tickers:
#     for interval in intervals:
#         data = yf.download(ticker, interval=interval, period=periods[intervals.index(interval)])
#         # Create a SQLite connection
#         conn = sqlite3.connect('index.db')
#         # Store the data into the SQLite database
#         data.to_sql(f'{ticker}_{interval}', conn, if_exists='replace')
#         # Close the SQLite connection
#         conn.close()

####################################################
####################################################
# 
# Combine the classes into one:
#   Stocks = StockData, Stocks, Stock
#     Static Methods: Create File, DB, Table
#     Private Methods: Get Data, Tickers, Intervals, Periods
#     Methods: Update Table Data, get period start from most recent 
#   Indicators = TechnicalIndicators, AdvancedTechnicalIndicators
#     Methods: Moving Average, Momentum, Stochastics, Standard Deviation
#     Methods: Ichimoku Cloud, Volume Weighted Average Price, Keltner Channel, Pivot Points, Volume, On Balance Volume, Accumulation Distribution Line, Money Flow Index, Commodity Channel Index, Rate of Change, True Strength Index, Know Sure Thing, Mass Index, Ulcer Index, Chaikin Oscillator, Detrended Price Oscillator, Ultimate Oscillator
#   BayesianProbabilities
#   StockPredictor
# 
####################################################
####################################################
# Class to fetch stock data and send it to the database
class StockData:
    def __init__(self, ticker, interval, period):
        self.ticker = ticker
        self.interval = interval
        self.period = period
        self.data = yf.download(ticker, interval=interval, period=period)
        self.conn = sqlite3.connect('index.db')
        self.data.to_sql(f'{ticker}_{interval}', self.conn, if_exists='replace')
        self.conn.close()

    def get_data(self):
        return self.data

    # def __getattribute__(self, __name: str) -> np.Any:
    #     return super().__getattribute__(__name)
    
    #  def __setattr__(self, __name: str, __value: np.Any) -> None:
    #      super().__setattr__(__name, __value)

    # Use Default Args in the __init__ method to set the default values for the ticker, interval and period
    def __get_data(self, ticker = 'AAPL', interval = '1d', period = '60d'):
        conn = sqlite3.connect('index.db')
        data = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', conn)
        conn.close()
        return data

    def __get_ticker(self):
        return self.ticker

    def __get_data_by_interval(self, interval):
        return self.data[interval]

    def __get_interval(self):
        return self.interval

    def __get_period(self):
        return self.period

    def __set_ticker(self, ticker):
        self.ticker = ticker

    def __set_interval(self, interval):
        self.interval = interval

    def __set_period(self, period):
        self.period = period

    def __str__(self):
        return f"Stock Data for {self.ticker} with interval {self.interval} and period {self.period}"
    
    def __repr__(self):
        return f"Stock Data for {self.ticker} with interval {self.interval} and period {self.period}"

# Class to fetch stock data for the list of tickers with the list of intervals, for either the max period, if there is no table data or the specified period if there is already data
class Stocks:
    def __init__(self, tickers, intervals, periods):
        self.tickers = tickers
        self.intervals = intervals
        self.periods = periods
        self.conn = sqlite3.connect('index.db')
        for ticker in tickers:
            for interval in intervals:
                if f'{ticker}_{interval}' not in pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ticker}_{interval}'", self.conn)['name'].values:
                    data = yf.download(ticker, interval=interval, period=periods[intervals.index(interval)])
                    data.to_sql(f'{ticker}_{interval}', self.conn, if_exists='replace')
        self.conn.close()

    def set_data(self, data):
        self.data = data

    def set_tickers(self, tickers):
        self.tickers = tickers

    def set_intervals(self, intervals):
        self.intervals = intervals

    def set_periods(self, periods):
        self.periods = periods

    def get_data(self):
        return self.data

    def get_tickers(self):
        return self.tickers

    def get_intervals(self):
        return self.intervals

    def get_periods(self):
        return self.periods

    def get_data_by_ticker(self, ticker):
        return self.data[ticker]
    
    def get_data_by_interval(self, interval):
        return self.data[interval]

    def show_tables(self):
        # Interact with the database class
        conn = sqlite3.connect('index.db')
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        conn.close()
        return tables

    def count_number_of_occurances(self, data, value):
        # Connect to the database and count the number of occurances of the 6% or greater increase in the stock price TSLA 1d
        conn = sqlite3.connect('index.db')
        # conditions = 
        data = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', conn)
        conn.close()
        return data[data == value].count()


class Stock:
    def __init__(self, ticker, interval):
        self.ticker = ticker
        self.interval = interval
        self.conn = sqlite3.connect('index.db')
        self.df = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', self.conn)
        self.conn.close()

    def get_data(self):
        return self.df

    def get_ticker(self):
        return self.ticker

    def get_interval(self):
        return self.interval

    def get_open(self):
        return self.df['Open']

    def get_high(self):
        return self.df['High']

    def get_low(self):
        return self.df['Low']

    def get_close(self):
        return self.df['Close']

    def get_volume(self):
        return self.df['Volume']
    
    def get_adj_close(self):
        return self.df['Adj Close']
    
    def get_date(self):
        return self.df.index

# Class for standard deviation, Momentum, Stochastics, Moving Averages
# Save to the database
# print("Technical Indicators")
# print("Moving Average", ti.get_moving_average(args.period)[-1:])
# print("Momentum", ti.get_momentum(args.period)[-1:])
# print("Stochastics", ti.get_stochastics(args.period)[-1:])
# print("Standard Deviation", ti.get_standard_deviation(args.period)[-1:])

# Document Code ^^
class TechnicalIndicators:
    def __init__(self, stock):
        self.stock = stock

    def get_moving_average(self, window):
        return self.stock.get_close().rolling(window=window).mean()

    def get_momentum(self, period):
        return self.stock.get_close() / self.stock.get_close().shift(period)

    def get_stochastics(self, period):
        return (self.stock.get_close() - self.stock.get_low().rolling(window=period).min()) / (self.stock.get_high().rolling(window=period).max() - self.stock.get_low().rolling(window=period).min())

    def get_standard_deviation(self, period):
        return self.stock.get_close().rolling(window=period).std()

    def get_standard_deviation_bands(self, period, std):
        return self.stock.get_close().rolling(window=period).mean() + std * self.stock.get_close().rolling(window=period).std(), self.stock.get_close().rolling(window=period).mean() - std * self.stock.get_close().rolling(window=period).std()


# print("Advanced Technical Indicators")
# print("Ichimoku Cloud", ati.get_ichimoku_cloud(args.period)[-1:])
# print("Volume Weighted Average Price", ati.get_volume_weighted_average_price(args.period)[-1:])
# print("Keltner Channel", ati.get_keltner_channel(args.period)[-1:])
# print("Pivot Points", ati.get_pivot_points(args.period)[-1:])
# print(tabulate(table, headers=["Indicator", "Value"]))
# Advanced Technical Indicators
class AdvancedTechnicalIndicators(TechnicalIndicators):
    def __init__(self, stock):
        super().__init__(stock)

    def get_ichimoku_cloud(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min()) / 2

    def get_volume_weighted_average_price(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min() + self.stock.get_close().rolling(window=period).mean()) / 3

    def get_keltner_channel(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min() + self.stock.get_close().rolling(window=period).mean()) / 3

    def get_pivot_points(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min() + self.stock.get_close().rolling(window=period).mean()) / 3

    def get_volume(self, period):
        return self.stock.get_volume().rolling(window=period).mean()

    def get_on_balance_volume(self, period):
        return self.stock.get_volume().rolling(window=period).mean()

    def get_accumulation_distribution_line(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min() + self.stock.get_close().rolling(window=period).mean()) / 3

    def get_money_flow_index(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min() + self.stock.get_close().rolling(window=period).mean()) / 3

    def get_commodity_channel_index(self, period):
        return (self.stock.get_high().rolling(window=period).max() + self.stock.get_low().rolling(window=period).min() + self.stock.get_close().rolling(window=period).mean()) / 3

    def get_rate_of_change(self, period):
        return self.stock.get_close() / self.stock.get_close().shift(period)

    def get_true_strength_index(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    
    def get_know_sure_thing(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    
    def get_mass_index(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    
    def get_ulcer_index(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    
    def get_chaikin_oscillator(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    
    def get_detrended_price_oscillator(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    
    def get_ultimate_oscillator(self, period):
        return (self.stock.get_close().rolling(window=period).mean() + self.stock.get_close().rolling(window=period).mean()) / 2
    

# Advanced Technical Indicators for Different Stocks with different intervals and different periods
class AdvancedTechnicalIndicatorsDifferentStocksDifferentIntervalsDifferentPeriods(TechnicalIndicators):
    def __init__(self, tickers, intervals, periods):
        self.tickers = tickers
        self.intervals = intervals
        self.periods = periods
        self.conn = sqlite3.connect('index.db')
        self.data = {}
        for ticker in tickers:
            for interval in intervals:
                self.data[f'{ticker}_{interval}'] = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', self.conn)
        self.conn.close()

        def __str__(self):
            return f"Advanced Technical Indicators for {self.tickers} with intervals {self.intervals} and periods {self.periods}"
        
        def __repr__(self):
            return f"Advanced Technical Indicators for {self.tickers} with intervals {self.intervals} and periods {self.periods}"
        
        def get_data(self):
            return self.data
        
        def set_data(self, data):
            self.data = data

        def get_tickers(self):
            return self.tickers
        
        def set_tickers(self, tickers):
            self.tickers = tickers

        def get_intervals(self):
            return self.intervals
        
        def set_intervals(self, intervals):
            self.intervals = intervals

        def get_periods(self):
            return self.periods
        
        def __set_periods(self, periods):
            self.periods = periods

        def __get_moving_average(self, ticker, interval, period):
            return self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()
        
        def __get_momentum(self, ticker, interval, period):
            return self.data[f'{ticker}_{interval}']['Close'] / self.data[f'{ticker}_{interval}']['Close'].shift(period)
        
        def __get_stochastics(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['Close'] - self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min()) / (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() - self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min())
        
        def __get_standard_deviation(self, ticker, interval, period):
            return self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).std()
        
        def __get_bollinger_bands(self, ticker, interval, period, std):
            return self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean() + std * self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).std(), self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean() - std * self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).std()
        
        # Get Technical Indicators
        def __get_ichimoku_cloud(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min()) / 2
        
        def __get_volume_weighted_average_price(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 3
        
        def __get_keltner_channel(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 3
        
        def __get_pivot_points(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 3
        
        def __get_volume(self, ticker, interval, period):
            return self.data[f'{ticker}_{interval}']['Volume'].rolling(window=period).mean()
        
        def __get_on_balance_volume(self, ticker, interval, period):
            return self.data[f'{ticker}_{interval}']['Volume'].rolling(window=period).mean()
        
        def __get_accumulation_distribution_line(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 3
        
        def __get_money_flow_index(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 3
        
        def __get_commodity_channel_index(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['High'].rolling(window=period).max() + self.data[f'{ticker}_{interval}']['Low'].rolling(window=period).min() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 3
        
        def __get_rate_of_change(self, ticker, interval, period):
            return self.data[f'{ticker}_{interval}']['Close'] / self.data[f'{ticker}_{interval}']['Close'].shift(period)
        
        def __get_true_strength_index(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 2
        
        def __get_know_sure_thing(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 2
        
        def __get_mass_index(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 2
        
        def __get_ulcer_index(self, ticker, interval, period):
            return (self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean() + self.data[f'{ticker}_{interval}']['Close'].rolling(window=period).mean()) / 2
    
# print(stock_predictor.get_ticker())
# print(stock_predictor.get_open())

##  X, Y = stock_predictor.preprocess_data() # Preprocess the data
##  x_train, x_test, y_train, y_test = stock_predictor.split_data(X, Y, test_size=0.2) # Split the data into training and testing sets
##  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshape the data for the LSTM model in the form of 
##  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))     # (samples, time steps, features) to fit the model input shape
## stock_predictor.build_model((x_train.shape[1], 1)) # Build the LSTM model
##    stock_predictor.train_model(x_train, y_train, batch_size=10, epochs=10) # Train the LSTM model
##    predictions = stock_predictor.predict(x_test) # Predict the stock prices
##    print(predictions[-2:])
##    print(stock_predictor.get_accuracy())
# Class For tensorflow model to predict stock prices using sqlite3 database
class StockPredictor(Stock):
    def __init__(self, ticker, interval, look_back=1):
        self.ticker = ticker
        self.interval = interval
        self.conn = sqlite3.connect('index.db')
        self.df = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', self.conn)
        self.conn.close()
        self.look_back = look_back
        self.model = None

    def preprocess_data(self):
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df['Close'].values.reshape(-1,1))

        # Create a time series dataset
        dataX, dataY = [], []
        for i in range(len(scaled_data)-self.look_back-1):
            dataX.append(scaled_data[i:(i+self.look_back), 0])
            dataY.append(scaled_data[i + self.look_back, 0])
        return np.array(dataX), np.array(dataY)

    def split_data(self, X, Y, test_size=0.2):
        train_size = int(len(X) * (1 - test_size))
        trainX, testX = X[0:train_size,:], X[train_size:len(X),:]
        trainY, testY = Y[0:train_size], Y[train_size:len(Y)]
        return trainX, testX, trainY, testY

    def save_model(self, filepath = "model.keras"):
        if self.model is None:
            raise Exception("You must build and train the model first")
        self.model.save(filepath)

    def load_model(self, filepath = "model.keras"):
        self.model = load_model(filepath)

    def get_data(self):
        return self.df

    def get_ticker(self):
        return self.ticker

    def get_interval(self):
        return self.interval

    def get_open(self):
        return self.df['Open']

    def get_high(self):
        return self.df['High']

    def get_low(self):
        return self.df['Low']

    def get_close(self):
        return self.df['Close']

    def get_volume(self):
        return self.df['Volume']
    
    def get_adj_close(self):
        return self.df['Adj Close']
    
    def get_date(self):
        return self.df.index

    def get_model(self):
        return self.model

    def train_model(self):
        data = self.get_close()
        X, Y = self.preprocess_data(data)
        trainX, testX, trainY, testY = self.split_data(X, Y)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        self.model = model

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def plot(self):
        pass

    def get_accuracy(self):
        pass

    def get_loss(self):
        pass

    def get_precision(self):
        pass

    def get_recall(self):
        pass

    def get_f1_score(self):
        f1 = 2 * (self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall())
        return f1

    def get_confusion_matrix(self):
        y_pred = self.model.predict(x_test)
        y_pred = (y_pred > 0.5)
        return confusion_matrix(y_test, y_pred)


    def get_classification_report(self):
        pass

    def get_feature_importance(self):
        pass

    def get_feature_selection(self):
        pass

    def get_hyperparameter_tuning(self):
        pass

    def get_ensemble_learning(self):
        # Bagging, Boosting, Stacking, Voting
        bag = BaggingClassifier()
        boost = AdaBoostClassifier()
        stack = StackingClassifier()
        vote = VotingClassifier()
        return bag, boost, stack, vote

    def get_boosting(self):
        pass

    def get_bagging(self):
        pass

    def get_stacking(self):
        pass

    def get_voting(self):
        pass

    def get_gradient_boosting(self):
        pass

    def get_adaboost(self):
        pass

    def get_xgboost(self):
        pass

    def get_lightgbm(self):
        pass

    def get_catboost(self):
        pass

    def get_neural_network(self):
        pass

    def get_recurrent_neural_network(self):
        pass

    def get_convolutional_neural_network(self):
        pass

    def get_long_short_term_memory(self):
        pass


# Class For tensorflow model to predict stock prices
class StockPredictorWithTensorFlow(StockPredictor):
    def __init__(self, ticker, interval):
        super().__init__(ticker, interval)
        self.model = None

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, x_train, y_train, batch_size=1, epochs=1):
        if self.model is None:
            raise Exception("You must build the model first")
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, x_test):
        if self.model is None:
            raise Exception("You must build and train the model first")
        return self.model.predict(x_test)

    def get_accuracy(self):
        if self.model is None:
            raise Exception("You must build and train the model first")
        return self.model.evaluate(x_test, y_test)
    
    def save_model(self, filepath = "model.keras"):
        if self.model is None:
            raise Exception("You must build and train the model first")
        self.model.save(filepath)

    def load_model(self, filepath = "model.keras"):
        self.model = load_model(filepath)

    def plot(self):
        if self.model is None:
            raise Exception("You must build and train the model first")
        predictions = self.model.predict(x_test)
        plt.plot(predictions, color='blue', label='Predicted Stock Price')
        plt.plot(y_test, color='red', label='Real Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig('stock_price_prediction.png')
        plt.show()
    

# Class To Calculate Bayesian Probabilities
class BayesianProbabilities:
    def __init__(self, stock):
        self.stock = stock
    
    def get_close(self):
        return self.stock.get_close()

    def get_standard_deviation(self, period):
        return self.stock.get_close().rolling(window=period).std()

    def get_bayesian_probability(self, period):
        return (self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period)
    
    def get_bayesian_probability_up(self, period):
        return (self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) > 0
    
    def get_bayesian_probability_down(self, period):
        return (self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) < 0
    
    def get_bayesian_probability_neutral(self, period):
        return (self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) == 0
    
    def get_bayesian_probability_up_count(self, period):
        return ((self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) > 0).sum()

    def get_bayesian_probability_down_count(self, period):
        return ((self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) < 0).sum()

    def get_bayesian_probability_neutral_count(self, period):
        return ((self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) == 0).sum()

    def get_bayesian_probability_up_percentage(self, period):
        return ((self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) > 0).sum() / len(self.stock.get_close())
    
    def get_bayesian_probability_down_percentage(self, period):
        return ((self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) < 0).sum() / len(self.stock.get_close())
    
    def get_bayesian_probability_neutral_percentage(self, period):
        return ((self.stock.get_close() - self.stock.get_close().shift(period)) / self.stock.get_close().shift(period) == 0).sum() / len(self.stock.get_close())
    
    def results(self):
        print("Bayesian Probabilities")
        print("Bayesian Probability", bp.get_bayesian_probability(10))
        print("Bayesian Probability Up", bp.get_bayesian_probability_up(10))
        print("Bayesian Probability Down", bp.get_bayesian_probability_down(10))
        print("Bayesian Probability Neutral", bp.get_bayesian_probability_neutral(10))
        print("Bayesian Probability Up Count", bp.get_bayesian_probability_up_count(10))
        print("Bayesian Probability Down Count", bp.get_bayesian_probability_down_count(10))
        print("Bayesian Probability Neutral Count", bp.get_bayesian_probability_neutral_count(10))
        print("Bayesian Probability Up Percentage", bp.get_bayesian_probability_up_percentage(10))
        print("Bayesian Probability Down Percentage", bp.get_bayesian_probability_down_percentage(10))
        print("Bayesian Probability Neutral Percentage", bp.get_bayesian_probability_neutral_percentage(10))


# print("Advanced Bayesian Probabilities")
# jNum00 = 10**1
# jNum01 = 10**2
# jNum10 = 10**3
# print("Bayesian Probability Up Close Given Up Close " + str(jNum01) + ": ", abp.get_bayesian_probability_up_close_given_up_close(jNum01))
# print("Bayesian Probability Down Close Given Down Close " + str(jNum01) + ": ", abp.get_bayesian_probability_down_close_given_down_close(jNum01))
# print("Bayesian 100: P(" + args.ticker + " Up | " + args.ticker + " Up & " + args.ticker01 + " Up)", abp.get_bayes_tsla_up_given_tsla_up_aapl_up(100))
# Advanced Bayesian Probabilities
class AdvancedBayesianProbabilities(BayesianProbabilities):
    def __init__(self, stock):
        super().__init__(stock)

    def get_bayesian_probability_up_close_given_up_close(self, period):
        data = self.stock.get_close()
        # up_close = data[data > data.shift(period)]
        # up_close_given_up_close = up_close[up_close.shift(period) > data.shift(2*period)]
        # Reindex the data instead of using shifting
        up_close = data[data > data.shift(period)].reset_index(drop=True)
        up_close_given_up_close = up_close[up_close > up_close.shift(period)].reset_index(drop=True)
        return len(up_close_given_up_close) / len(up_close)

    # P( TSLA Up | TSLA Up & AAPL Up)
    def get_bayes_tsla_up_given_tsla_up_aapl_up(self, period):
        tsla = Stock(args.ticker, args.interval)
        aapl = Stock(args.ticker, args.interval)
        data = tsla.get_close()
        up_tsla = data[data > data.shift(period)].reset_index(drop=True)
        up_tsla_given_up_aapl = up_tsla[up_tsla > up_tsla.shift(period)].reset_index(drop=True)
        return len(up_tsla_given_up_aapl) / len(up_tsla)

    # P( Tsla > stdev | close last 10 days > stdev)
    def get_bayes_tsla_gt_stdev_given_close_last_10_days_gt_stdev(self, period):
        tsla = Stock(args.ticker, args.interval)
        data = tsla.get_close()
        stdev = data.rolling(window=period).std()
        close_last_10_days = data[-10:]
        close_last_10_days_gt_stdev = close_last_10_days[close_last_10_days > stdev]
        return len(close_last_10_days_gt_stdev) / len(close_last_10_days)

    def get_bayesian_probability_down_close_given_down_close(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)].reset_index(drop=True)
        down_close_given_down_close = down_close[down_close < down_close.shift(period)].reset_index(drop=True)
        return len(down_close_given_down_close) / len(down_close)
    
    def get_bayesian_probability_neutral_close_given_neutral_close(self, period):
        data = self.stock.get_close()
        neutral_close = data[data == data.shift(period)]
        neutral_close_given_neutral_close = neutral_close[neutral_close.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_neutral_close) / len(neutral_close)
    
    def get_bayesian_probability_up_close_given_ten_up_closes(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)].reset_index(drop=True)
        for i in range(1, 11):
            up_close = up_close[up_close.shift(i) > data.shift(i+period)].reset_index(drop=True)
        return len(up_close) / len(data)
    
    def get_bayesian_probability_down_close_given_mostly_up_closes_in_last_10_periods(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)].reset_index(drop=True)
        up_close_given_up_close = up_close[up_close.shift(period) > data.shift(2*period)]
        mostly_up_closes_in_last_10_periods = up_close_given_up_close[up_close_given_up_close.shift(period) > data.shift(3*period)]
        return len(mostly_up_closes_in_last_10_periods) / len(up_close_given_up_close)
    
    def get_bayesian_probability_down_close_given_mostly_down_closes_in_last_10_periods(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)].reset_index(drop=True)
        down_close_given_down_close = down_close[down_close.shift(period) < data.shift(2*period)]
        mostly_down_closes_in_last_10_periods = down_close_given_down_close[down_close_given_down_close.shift(period) < data.shift(3*period)]
        return len(mostly_down_closes_in_last_10_periods) / len(down_close_given_down_close)
    
    def get_bayesian_probability_up_close_given_mostly_up_closes_in_last_10_periods(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)].reset_index(drop=True)
        up_close_given_up_close = up_close[up_close.shift(period) > data.shift(2*period)]
        mostly_up_closes_in_last_10_periods = up_close_given_up_close[up_close_given_up_close.shift(period) > data.shift(3*period)]

    def get_bayesian_probability_up_close_given_5_of_last_10_periods_up_close_or_1_of_last_2_periods_up_close(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)].reset_index(drop=True)
        up_close_given_up_close = up_close[up_close.shift(period).reset_index(drop=True) > data.shift(2*period)].reset_index(drop=True)
        mostly_up_closes_in_last_10_periods = up_close_given_up_close[up_close_given_up_close.shift(period).reset_index(drop=True) > data.shift(3*period).reset_index(drop=True)].reset_index(drop=True)
        return len(mostly_up_closes_in_last_10_periods) / len(up_close_given_up_close)
    
    def get_bayesian_probability_up_close_given_down_close(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)]
        up_close_given_down_close = down_close[down_close.shift(period) > data.shift(2*period)]
        return len(up_close_given_down_close) / len(down_close)
    
    def get_bayesian_probability_down_close_given_up_close(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)]
        down_close_given_up_close = up_close[up_close.shift(period) < data.shift(2*period)]
        return len(down_close_given_up_close) / len(up_close)

    def get_bayesian_probability_up_close_given_mostly_down_closes_in_last_10_periods(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)].reset_index(drop=True)
        down_close_given_down_close = down_close[down_close.shift(period) < data.shift(2*period)]
        mostly_down_closes_in_last_10_periods = down_close_given_down_close[down_close_given_down_close.shift(period) < data.shift(3*period)]
        return len(mostly_down_closes_in_last_10_periods) / len(down_close_given_down_close)

    def get_bayesian_probability_down_close_given_ten_down_closes(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)].reset_index(drop=True)
        for i in range(1, 11):
            down_close = down_close[down_close.shift(i) < data.shift(i+period)].reset_index(drop=True)
        return len(down_close) / len(data)

    # Given 2 previous periods
    def get_bayesian_probability_neutral_close_given_up_close_up_close(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)]
        up_close_given_up_close = up_close[up_close.shift(period) > data.shift(2*period)]
        neutral_close_given_up_close_up_close = up_close_given_up_close[up_close_given_up_close.shift(period) == data.shift(3*period)]
        return len(neutral_close_given_up_close_up_close) / len(up_close_given_up_close)
    
    def get_bayesian_probability_neutral_close_given_down_close_down_close(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)]
        down_close_given_down_close = down_close[down_close.shift(period) < data.shift(2*period)]
        neutral_close_given_down_close_down_close = down_close_given_down_close[down_close_given_down_close.shift(period) == data.shift(3*period)]
        return len(neutral_close_given_down_close_down_close) / len(down_close_given_down_close)
    
    # Given 10 previous periods
    def get_bayesian_probability_neutral_close_given_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)]
        up_close_given_up_close = up_close[up_close.shift(period) > data.shift(2*period)]
        neutral_close_given_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close = up_close_given_up_close[up_close_given_up_close.shift(period) == data.shift(11*period)]
        return len(neutral_close_given_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close_up_close) / len(up_close_given_up_close)
    
    
    def get_bayesian_probability_neutral_close_given_up_close(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)]
        neutral_close_given_up_close = up_close[up_close.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_up_close) / len(up_close)
    
    def get_bayesian_probability_neutral_close_given_down_close(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)]
        neutral_close_given_down_close = down_close[down_close.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_down_close) / len(down_close)
    
    def get_bayesian_probability_up_close_given_neutral_close(self, period):
        data = self.stock.get_close()
        neutral_close = data[data == data.shift(period)]
        up_close_given_neutral_close = neutral_close[neutral_close.shift(period) > data.shift(2*period)]
        return len(up_close_given_neutral_close) / len(neutral_close)
    
    def get_bayesian_probability_down_close_given_neutral_close(self, period):
        data = self.stock.get_close()
        neutral_close = data[data == data.shift(period)]
        down_close_given_neutral_close = neutral_close[neutral_close.shift(period) < data.shift(2*period)]
        return len(down_close_given_neutral_close) / len(neutral_close)
    
    def get_bayesian_probability_neutral_close_given_up_close(self, period):
        data = self.stock.get_close()
        up_close = data[data > data.shift(period)]
        neutral_close_given_up_close = up_close[up_close.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_up_close) / len(up_close)

    def get_bayesian_probability_neutral_close_given_down_close(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)]
        neutral_close_given_down_close = down_close[down_close.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_down_close) / len(down_close)
    
    def get_bayesian_probability_up_close_given_up_high(self, period):
        data = self.stock.get_high()
        up_high = data[data > data.shift(period)]
        up_close_given_up_high = up_high[up_high.shift(period) > data.shift(2*period)]
        return len(up_close_given_up_high) / len(up_high)
    
    def get_bayesian_probability_down_close_given_down_high(self, period):
        data = self.stock.get_high()
        down_high = data[data < data.shift(period)]
        down_close_given_down_high = down_high[down_high.shift(period) < data.shift(2*period)]
        return len(down_close_given_down_high) / len(down_high)
    
    def get_bayesian_probability_neutral_close_given_neutral_high(self, period):
        data = self.stock.get_high()
        neutral_high = data[data == data.shift(period)]
        neutral_close_given_neutral_high = neutral_high[neutral_high.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_neutral_high) / len(neutral_high)
    
    def get_bayesian_probability_up_close_given_down_high(self, period):
        data = self.stock.get_high()
        down_high = data[data < data.shift(period)]
        up_close_given_down_high = down_high[down_high.shift(period) > data.shift(2*period)]
        return len(up_close_given_down_high) / len(down_high)
    
    def get_bayesian_probability_down_close_given_up_high(self, period):
        data = self.stock.get_high()
        up_high = data[data > data.shift(period)]
        down_close_given_up_high = up_high[up_high.shift(period) < data.shift(2*period)]
        return len(down_close_given_up_high) / len(up_high)
    
    def get_bayesian_probability_neutral_close_given_up_high(self, period):
        data = self.stock.get_high()
        up_high = data[data > data.shift(period)]
        neutral_close_given_up_high = up_high[up_high.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_up_high) / len(up_high)
    
    def get_bayesian_probability_neutral_close_given_down_high(self, period):
        data = self.stock.get_high()
        down_high = data[data < data.shift(period)]
        neutral_close_given_down_high = down_high[down_high.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_down_high) / len(down_high)
    
    def get_bayesian_probability_up_close_given_neutral_high(self, period):
        data = self.stock.get_high()
        neutral_high = data[data == data.shift(period)]
        up_close_given_neutral_high = neutral_high[neutral_high.shift(period) > data.shift(2*period)]
        return len(up_close_given_neutral_high) / len(neutral_high)
    
    def get_bayesian_probability_down_close_given_neutral_high(self, period):
        data = self.stock.get_high()
        neutral_high = data[data == data.shift(period)]
        down_close_given_neutral_high = neutral_high[neutral_high.shift(period) < data.shift(2*period)]
        return len(down_close_given_neutral_high) / len(neutral_high)
    
    def get_bayesian_probability_neutral_close_given_up_high(self, period):
        data = self.stock.get_high()
        up_high = data[data > data.shift(period)]
        neutral_close_given_up_high = up_high[up_high.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_up_high) / len(up_high)
    
    def get_bayesian_probability_neutral_close_given_down_high(self, period):
        data = self.stock.get_high()
        down_high = data[data < data.shift(period)]
        neutral_close_given_down_high = down_high[down_high.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_down_high) / len(down_high)
    
    def get_bayesian_probability_up_close_given_up_low(self, period):
        data = self.stock.get_low()
        up_low = data[data > data.shift(period)]
        up_close_given_up_low = up_low[up_low.shift(period) > data.shift(2*period)]
        return len(up_close_given_up_low) / len(up_low)
    
    def get_bayesian_probability_down_close_given_down_low(self, period):
        data = self.stock.get_low()
        down_low = data[data < data.shift(period)]
        down_close_given_down_low = down_low[down_low.shift(period) < data.shift(2*period)]
        return len(down_close_given_down_low) / len(down_low)

    def get_bayesian_probability_neutral_close_given_neutral_low(self, period):
        data = self.stock.get_low()
        neutral_low = data[data == data.shift(period)]
        neutral_close_given_neutral_low = neutral_low[neutral_low.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_neutral_low) / len(neutral_low)

    def get_bayesian_probability_up_close_given_down_low(self, period):
        data = self.stock.get_low()
        down_low = data[data < data.shift(period)]
        up_close_given_down_low = down_low[down_low.shift(period) > data.shift(2*period)]
        return len(up_close_given_down_low) / len(down_low)

    def get_bayesian_probability_down_close_given_up_low(self, period):
        data = self.stock.get_low()
        up_low = data[data > data.shift(period)]
        down_close_given_up_low = up_low[up_low.shift(period) < data.shift(2*period)]
        return len(down_close_given_up_low) / len(up_low)

    def get_bayesian_probability_neutral_close_given_up_low(self, period):
        data = self.stock.get_low()
        up_low = data[data > data.shift(period)]
        neutral_close_given_up_low = up_low[up_low.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_up_low) / len(up_low)

    def get_bayesian_probability_neutral_close_given_down_low(self, period):
        data = self.stock.get_low()
        down_low = data[data < data.shift(period)]
        neutral_close_given_down_low = down_low[down_low.shift(period) == data.shift(2*period)]
        return len(neutral_close_given_down_low) / len(down_low)

    def get_bayesian_probability_up_close_given_neutral_low(self, period):
        data = self.stock.get_low()
        neutral_low = data[data == data.shift(period)]
        up_close_given_neutral_low = neutral_low[neutral_low.shift(period) > data.shift(2*period)]
        return len(up_close_given_neutral_low) / len(neutral_low)

    def get_bayesian_probability_down_close_given_neutral_low(self, period):
        data = self.stock.get_low()
        neutral_low = data[data == data.shift(period)]
        down_close_given_neutral_low = neutral_low[neutral_low.shift(period) < data.shift(2*period)]
        return len(down_close_given_neutral_low) / len(neutral_low)


    def get_bayesian_probability_neutral_close_given_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close(self, period):
        data = self.stock.get_close()
        down_close = data[data < data.shift(period)]
        down_close_given_down_close = down_close[down_close.shift(period) < data.shift(2*period)]
        neutral_close_given_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close = down_close_given_down_close[down_close_given_down_close.shift(period) == data.shift(11*period)]
        return len(neutral_close_given_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close_down_close / len(down_close_given_down_close))

    def get_bayesian_probability_neutral_close_given_ten_neutral_closes(self, period):
        data = self.stock.get_close()
        neutral_close = data[data == data.shift(period)]
        for i in range(1, 11):
            neutral_close = neutral_close[neutral_close.shift(i) == data.shift(i+period)]
        return len(neutral_close) / len(data)
    
    # Given 100 previous periods
    # def get_bayesian_probability_neutral_close_given_down_close

    # P(TSLA Up | AAPL And AMZN Up)
    # P(TSLA Up | AAPL And AMZN Down)
    # P(TSLA Up | AAPL And AMZN Neutral)
    # P(TSLA Up | AAPL Up And AMZN Down)
    # P(TSLA Up | AAPL Up And AMZN Neutral)
    # P(TSLA Up | AAPL Down And AMZN Up)
    # P(TSLA Up | AAPL Down And AMZN Down)
    # P(TSLA Up | AAPL Down And AMZN Neutral)
    # P(TSLA Up | AAPL Neutral And AMZN Up)
    # P(TSLA Up | AAPL Neutral And AMZN Down) or P(TSLA Up | AAPL Neutral And AMZN Neutral) or P(TSLA Up | AAPL Neutral And AMZN Neutral)
    # P(TSLA Up | AAPL Up And AMZN Up) or P(TSLA Up | AAPL Up And AMZN Down) or P(TSLA Up | AAPL Up And AMZN Neutral)
    # P(TSLA Up | AAPL Down And AMZN Up) or P(TSLA Up | AAPL Down And AMZN Down) or P(TSLA Up | AAPL Down And AMZN Neutral)
    # P(TSLA Up | AAPL Neutral And AMZN Up) or P(TSLA Up | AAPL Neutral And AMZN Down) or P(TSLA Up | AAPL Neutral And AMZN Neutral)

    # P(up | momentum, stochastics, moving average, standard deviation, bollinger bands, relative strength index, average directional index, trix, macd, aroon)
    # P(down | momentum, stochastics, moving average, standard deviation, bollinger bands, relative strength index, average directional index, trix, macd, aroon)
    # P(neutral | momentum, stochastics, moving average, standard deviation, bollinger bands, relative strength index, average directional index, trix, macd, aroon)
    # P(Up | Momentum Up, Stochastics Up, Moving Average Up, Standard Deviation Up, Bollinger Bands Up, Relative Strength Index Up, Average Directional Index Up, Trix Up, Macd Up, Aroon Up)
    # P(Down | Momentum Down, Stochastics Down, Moving Average Down, Standard Deviation Down, Bollinger Bands Down, Relative Strength Index Down, Average Directional Index Down, Trix Down, Macd Down, Aroon Down)
    # P(Up | Momentum Down, Stochastics Down, Moving Average Down, Standard Deviation Down, Bollinger Bands Down, Relative Strength Index Down, Average Directional Index Down, Trix Down, Macd Down, Aroon Down)

class Bayes(Stock):
    def __init__(self, ticker, interval):
        self.ticker = ticker
        self.interval = interval
        self.conn = sqlite3.connect('index.db')
        self.df = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', self.conn)
        self.conn.close()

    def get_data(self):
        return self.df

# Take arbitrary number of tickers and intervals however the number of tickers and intervals must be the same
class Bayes007(Stock):
    def __init__(self, tickers, intervals):
        self.tickers = tickers
        self.intervals = intervals
        self.conn = sqlite3.connect('index.db')
        self.dfs = []
        for ticker, interval in zip(tickers, intervals):
            self.dfs.append(pd.read_sql(f'SELECT * FROM {ticker}_{interval}', self.conn))
        self.conn.close()

    def get_data(self):
        return self.dfs

    # P(ticker1 Up | ticker2 Up, ticker3 Up)

    # Std Dev of the stock prices
    def get_standard_deviation(self, period):
        return self.get_close().rolling(window=period).std()

    def get_moving_average(self, period):
        return self.get_close().rolling(window=period).mean()
    
    # Bollinger Bands
    def get_bollinger_bands(self, period, std_dev):
        return self.get_moving_average(period) + std_dev * self.get_standard_deviation(period), self.get_moving_average(period) - std_dev * self.get_standard_deviation(period)
    
    # Return P of current stock price, given standard deviation
    def get_bayesian_probability_up(self, period):
        up_probabilities = []
        for df in self.dfs:
            data = df['Close']
            up_close = data[data > data.shift(period)]
            up_close_given_previous_up_close = up_close[up_close.shift(period) > data.shift(2*period)]
            up_probabilities.append(len(up_close_given_previous_up_close) / len(up_close))
        return up_probabilities

# Class for scraping stock data
class TickerSymbolScraper:
    def __init__(self, url, ticker = "TSLA"):
        self.url = url
        self.__ticker = ticker

    def run_multiprocesses(self, symbols):
        def run_process(symbol):
            # Your subprocess call
            if not os.path.exists(f'{symbol}.db'):
                Stock(symbol, args).create_table()
            subprocess.Popen(["python", "index.py", "--ticker", symbol, "--interval", "1d"])
        
        # Create tasks for each symbol
        with multiprocessing.Pool() as pool:
            pool.map(run_process, symbols)

    async def run_async_processes(self, symbols):
        async def run_process(symbol):
            # Your subprocess call
            subprocess.Popen(["python", "index.py", "--ticker", symbol])
            subprocess.Popen(["python", "index.py", "--ticker", symbol, "--interval", "1d"])

        # Create tasks for each symbol
        tasks = [run_process(symbol) for symbol in symbols]

        # Run the tasks concurrently
        await asyncio.gather(*tasks)

    def rs_data(self):
        r = requests.get(self.url)
        return BeautifulSoup(r.text, 'html.parser')

    def __data_searches__(self, searches = []):
        rs = self.rs_data()
        data = []
        for search in searches:
            for link in rs.find_all(lambda tag: tag.get(search) is not None):
                data.append(link.get(search))
        return data

    def _names_and_prices(self):
        rs = self.rs_data()
        names_and_prices = []
        for link in rs.find_all(lambda tag: tag.get('data-name') is not None):
            names_and_prices.append((link.get('data-name'), link.get('data-price')))
        return names_and_prices

    def __names_and_prices__(self):
        name = self.__data_searches__(['data-price'])
        price = self.__data_searches__(['data-name'])
        return list(zip(name, price))

    def tkr_title(self):
        rs = self.rs_data()
        tkr = []
        for link in rs.find_all('title'):
            tkr.append(link.get_text())
        return tkr

    def prices(self):
        rs = self.rs_data()
        prices = []
        for l in rs.find_all(lambda tag: tag.get('data-price') is not None):
            prices.append(l['data-price'])
        return prices

    def names(self):
        rs = self.rs_data()
        names = []
        for link in rs.find_all(lambda tag: tag.get('data-name') is not None):
            names.append(link.get('data-name'))
        return names

    def get_data_attributes(self):
        rs = self.rs_data()
        data_attributes = []
        for tag in rs.find_all(True):
            for attr in tag.attrs:
                data_attributes.append(attr)
        return data_attributes

    def __hiJinx__(self):
        rs = self.rs_data()
        hiJinx = []
        search = 'data-*'
        for link in rs.find_all(lambda tag: tag.get(search) is not None):
            hiJinx.append(link.get(search))
        return hiJinx   

    def get_ticker_symbols(self):
        rs = self.rs_data()
        ticker_symbols = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if "/quote/" in href:
              ticker_symbols.append(href.split("/quote/")[1])

        return ticker_symbols

    def get_ticker_symbol(self, index):
        return self.get_ticker_symbols()[index]
    
    def __iter__(self):
        return self.get_ticker_symbols().__iter__()
    
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class Database:
    def __init__(self, ticker, interval):
      self.ticker = ticker
      self.interval = interval
      self.conn = sqlite3.connect('index.db')
      self.df = pd.read_sql(f'SELECT * FROM {ticker}_{interval}', self.conn)
      self.conn.close()

    def get_data(self):
      return self.df

    def set_data(self, df):
      self.df = df

# class Database007 {
#     constructor(dbFilePath) {
#         this.db = new sqlite3.Database(dbFilePath, (err) => {
#             if (err) {
#                 console.log('Could not connect to database', err);
#             } else {
#                 console.log('Connected to database');
#             }
#         });
#     }
# 
#     run(sql, params = []) {
#         return new Promise((resolve, reject) => {
#             this.db.run(sql, params, function (err) {
#                 if (err) {
#                     console.log('Error running sql ' + sql);
#                     console.log(err);
#                     reject(err);
#                 } else {
#                     resolve({ id: this.lastID });
#                 }
#             });
#         });
#     }
# 
#     get(sql, params = []) {
#         return new Promise((resolve, reject) => {
#             this.db.get(sql, params, (err, result) => {
#                 if (err) {
#                     console.log('Error running sql: ' + sql);
#                     console.log(err);
#                     reject(err);
#                 } else {
#                     resolve(result);
#                 }
#             });
#         });
#     }
# 
#     all(sql, params = []) {
#         return new Promise((resolve, reject) => {
#             this.db.all(sql, params, (err, rows) => {
#                 if (err) {
#                     console.log('Error running sql: ' + sql);
#                     console.log(err);
#                     reject(err);
#                 } else {
#                     resolve(rows);
#                 }
#             });
#         });
#     }
# 
#     delete(sql, params = []) {
#         return new Promise((resolve, reject) => {
#             this.db.run(sql, params, function (err) {
#                 if (err) {
#                     console.log('Error running sql ' + sql);
#                     console.log(err);
#                     reject(err);
#                 } else {
#                     resolve({ rowsDeleted: this.changes });
#                 }
#             });
#         });
#     }
# }
# 
# module.exports = Database;


if __name__ == '__main__':
    __tickers = ['TSLA', 'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'META']
    __cryptos = ['BTC', 'ETH']
    __indices = ['DJI', 'IXIC', 'INX']
    __intervals = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '5d', '1mo', '3mo']
    __periods = ['7d', '60d', '60d', '60d', '60d', '60d', "max", 'max', 'max', 'max']

    
    parser = argparse.ArgumentParser(description="Stock Market Analysis")
    parser.add_argument("-T", "--ticker", type=str, help="Stock Ticker", default=os.getenv("TICKER"))
    parser.add_argument("-C", "--crypto", type=str, help="Crypto Ticker", default=__cryptos[0])
    parser.add_argument("-I", "--index", type=str, help="Index Ticker", default=__indices[0])
    parser.add_argument("--ticker01", type=str, help="Stock Ticker", default=__tickers[1])
    parser.add_argument("-i", "--interval", type=str, help="Stock Interval", default=__intervals[0])
    # parser.add_argument("-o", "--option", type=str, help="Option", default="tickers")
    parser.add_argument("-p", "--period", type=int, help="Stock Period", default=20)
    parser.add_argument("-l", "--look-back", type=int, help="Look Back Period", default=60)
    parser.add_argument("-s", "--stdev", type=int, help="Standard Deviation", default=20)
    parser.add_argument("-m", "--moving-average", type=int, help="Moving Average", default=20)
    parser.add_argument("-M", "--momentum", type=int, help="Momentum", default=20)
    parser.add_argument("-S", "--stochastics", type=int, help="Stochastics", default=20)
    parser.add_argument("-o", "--model", type=str, help="Model", default="LSTM")
    parser.add_argument_group("Technical Indicators")
    args = parser.parse_args()

    url = "https://www.google.com/finance/quote/" 
    # If -C --crypto is set use __cryptos elif -I or --indicies is set use __indices else use __tickers
    if args.crypto:
        __d = __cryptos
        _d = args.crypto
    elif args.index:
        __d = __indices
        _d = args.indicies
    else:
        __d = __tickers 
        _d = args.tickers

    stocks = Stocks(__d, __intervals, __periods)
    stock = Stock(args.ticker, args.interval)
    ti = TechnicalIndicators(stock)
    ati = AdvancedTechnicalIndicators(stock) # Advanced Technical Indicators
    bp = BayesianProbabilities(stock) # Bayesian Probabilities
    abp = AdvancedBayesianProbabilities(bp) # Advanced Bayesian Probabilities

    ##  stock_predictor_wout_tensorflow = StockPredictor(args.ticker, args.interval, args.look_back)
    ##  stock_predictor = StockPredictorWithTensorFlow(stock_predictor_wout_tensorflow)


    if args.ticker in __indices:
        url += args.ticker + ":INDEXDJX"
    elif args.ticker in __cryptos:
        url += args.ticker + "-USD"
    else:
        url += args.ticker + ":NASDAQ"

    scraper = TickerSymbolScraper(url, args.ticker)
    # If option -j is used, print hiJinx
    try:
        print("Stock Data" + _d + " " + args.interval)
        symbols = scraper.__data_searches__(['data-symbol'])
        # asyncio.run(scraper.run_async_processes(symbols))

        # print(args.ticker + " Last Price: ", scraper.__data_searches__(['data-last-price']))
        print(symbols, scraper.run_multiprocesses(symbols))
        print("Names and Prices: ", scraper.__names_and_prices__())
        print("Symbols: ", symbols)
        # for symbol in symbols:
        #     sleep(60)
        #     subprocess.run(["python", "index.py", "--ticker", symbol])

    except Exception as e:
        print("Error")
        # Switch Case Any a for all data or e for error report
        choice = input("Enter a for all data or e for error report: ")
        if choice == 'a':
            print(scraper.rs_data())
        elif choice == 'w':
            print("")
        else:
            print(e)
