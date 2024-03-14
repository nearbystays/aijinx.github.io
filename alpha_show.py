import sqlite3
import pandas as pd
from datetime import date

def show_data():
    # Create a connection to the SQLite3 database
    conn = sqlite3.connect('tsla.db')

    # Query the data from the 'tsla' table for today's date
    query = f"SELECT * FROM tsla WHERE date = '{date.today()}'"
    data = pd.read_sql_query(query, conn)

    # Print the data
    print(data)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    show_data()