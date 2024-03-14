# bash function using alphavantage get 1 minute data for max historical period and store to sqlite3 file named tsla.db database tsla table date using jinx.sh
#!/bin/bash

alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY"

# Create a function to create the table
function create_table {
    sqlite3 tsla.db "CREATE TABLE IF NOT EXISTS tsla (date TEXT, time TEXT, price REAL, after_hours REAL, interval TEXT, volume INTEGER);"
}

# Get the data for tsla for 1 minute interval
curl -o tsla_1m.json "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&apikey=$alpha_vantage_key"

# Create a function to insert data into the table
function insert_data {
    # Get the date, time, price, after hours, interval, and volume from the json file
    date=$(jq -r '.date' tsla_1m.json)
    time=$(jq -r '.time' tsla_1m.json)
    price=$(jq -r '."1. open"' tsla_1m.json)
    after_hours=$(jq -r '."4. close"' tsla_1m.json)
    interval="1m"
    volume=$(jq -r '."5. volume"' tsla_1m.json)

    # Insert the data into the table
    sqlite3 tsla.db "INSERT INTO tsla (date, time, price, after_hours, interval, volume) VALUES ('$date', '$time', $price, $after_hours, '$interval', $volume);"
}

create_table
insert_data