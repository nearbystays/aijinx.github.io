import pandas as pd
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt

# Get the data for the percent gainers
gainers = si.get_day_gainers()

# Convert to DataFrame
df = pd.DataFrame(gainers)

# Convert the 'Change %' column to numeric values and filter rows where 'Change pct' is over 100
df['Change pct'] = df['Change pct'].str.rstrip('%').astype('float')
df_over_100 = df[df['Change pct'] > 100]

# Plot the 'Change pct' of the stocks that gained over 100pct
df_over_100.plot(kind='bar', x='Symbol', y='Change pct', legend=False)
plt.title('Stocks with over 100pct gain')
plt.ylabel('Change pct')
plt.show()