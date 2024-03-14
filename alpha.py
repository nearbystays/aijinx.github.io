from alpha_vantage.timeseries import TimeSeries
import sqlite3
import pandas as pd

def get_data():
    # Initialize the TimeSeries class with your Alpha Vantage API key
    ts = TimeSeries(key='YOUR_ALPHA_VANTAGE_API_KEY', output_format='pandas')

    # Get 1-minute data for the maximum historical period
    data, meta_data = ts.get_intraday(symbol='TSLA', interval='1min', outputsize='full')

    # Create a connection to the SQLite3 database
    conn = sqlite3.connect('tsla.db')

    # Write the data to the 'tsla' table in the 'tsla' database
    data.to_sql('tsla', conn, if_exists='replace')

    # Close the connection
    conn.close()

if __name__ == "__main__":
    get_data()