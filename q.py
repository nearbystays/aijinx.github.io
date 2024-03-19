import sqlite3
import pandas as pd

# Create a SQLite connection
conn = sqlite3.connect('index.db')

# Query the data
df_1m = pd.read_sql_query("SELECT * from TSLA_1m", conn)
df_1h = pd.read_sql_query("SELECT * from TSLA_1h", conn)
df_1d = pd.read_sql_query("SELECT * from TSLA_1d", conn)
df_1w = pd.read_sql_query("SELECT * from TSLA_1w", conn)

# Close the SQLite connection
conn.close()

# Print the data
print(df_1m)
print(df_1h)
print(df_1d)
print(df_1w)
