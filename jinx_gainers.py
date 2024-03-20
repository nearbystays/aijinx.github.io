from yahoo_fin import stock_info as si
from yahoo_fin import options
import pandas as pd

# Get the data for the percent gainers
gainers = si.get_data('gainers')

# Print the data
#Convert to table format using pandas
df = pd.DataFrame(gainers)
# Count
print(df.head(10))
print(df.tail(10))
print(df.count())