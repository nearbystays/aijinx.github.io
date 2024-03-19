import hashlib

# The string to hash
s = "Hello, World!"

# Create a new SHA-256 hash object
hash_object = hashlib.sha256()

# Update the hash object with the bytes of the string
hash_object.update(s.encode())

# Get the hexadecimal representation of the hash
hex_dig = hash_object.hexdigest()

print(hex_dig)

# List of miners
miners = [
    {'hash_rate': 19.5, 'power_consumption': 2000},
    {'hash_rate': 14, 'power_consumption': 1375},
    # Add more miners as needed
]

# Get the current Bitcoin price
response = requests.get('https://api.coindesk.com/v1/bpi/currentprice/BTC.json')
btc_price = response.json()['bpi']['USD']['rate_float']

# hash_rate = float(19.5 * 1000000000) # 1 TH/s = 1,000,000,000,000 hashes per second # Get the hash rate from the Antminer S9 currently at 19.5 TH/s keep in TH/s
# power_consumption_kw = float(1800 / 1000) # Power consumption in kilowatts

cost = [0.15, 0.25, 0.35]
for miner in miners:
    for i in range(3):
        # Cost per kilowatt-hour in dollars
        cost_per_kwh = cost[i]

        # Calculate the daily electricity cost
        daily_electricity_cost = miner['power_consumption'] / 1E3 * cost_per_kwh * 24

        # Get the current Bitcoin difficulty
        response = requests.get('https://blockchain.info/q/getdifficulty')
        difficulty = float(response.text)

        # print('Current Bitcoin difficulty: ', difficulty)

        # Calculate the daily earnings
        daily_earnings = (btc_price * miner['hash_rate'] * 10E8) / difficulty

        # Calculate the daily profit
        daily_profit = daily_earnings - daily_electricity_cost

        print('Cost Per KWH: ', cost_per_kwh, 'USD')
        print(f'Daily Electricity: ', daily_electricity_cost, 'USD')
        print(f'Daily Earnings: ', daily_profit, 'USD')
        print(f'Monthly Profilt: ', daily_profit * 30, 'USD')
        print(f'Difficulty: ', difficulty)
        print('')