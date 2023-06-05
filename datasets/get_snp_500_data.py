import yfinance as yf
import pandas as pd
import datetime
from tqdm import tqdm


# Get the list of symbols
end_date = datetime.date.today()
symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH', 'BAC', 'HD', 'DIS', 'VZ', 'NFLX', 'ADBE', 'PYPL', 'CRM', 'KO', 'INTC', 'CSCO', 'PFE', 'CMCSA', 'XOM', 'T', 'COST', 'PEP', 'BA', 'NVDA', 'ABT', 'CVX', 'TMO', 'MRK', 'ABBV', 'WMT', 'MCD', 'C', 'NKE', 'ACN', 'UNP', 'HON', 'ORCL', 'IBM', 'NEE', 'LLY', 'TXN', 'MMM', 'AMGN', 'DHR', 'AVGO', 'COP', 'LMT', 'SPG', 'AXP', 'MDT', 'NVS', 'LIN', 'CAT', 'SBUX', 'ADP', 'SCHW', 'GS', 'DE', 'BLK', 'HD', 'BMY', 'CI', 'BDX', 'RTX', 'D', 'DUK', 'NEE', 'WM', 'BDX', 'PLD', 'PNC', 'FIS', 'LOW', 'BDX', 'CNC', 'COF', 'GE', 'DOW', 'BK', 'CCL', 'CMA', 'CB', 'APD', 'SYY', 'SRE', 'PGR', 'PEG', 'GLW', 'DD', 'BAX', 'BLL', 'F', 'WBA', 'MET', 'MMC', 'CBRE', 'VLO', 'MMC', 'CMI', 'HON', 'AON', 'BLK', 'CB', 'CARR', 'HIG', 'TSN', 'AEP', 'AIG', 'LIN', 'WFC', 'KR', 'DTE', 'COF', 'PCAR', 'CTAS', 'MCK', 'HCA', 'ZTS', 'ZBH', 'ZION', 'ZBRA']
start_date = '2010-01-01'

all_data = []
# Calculate the number of years between start_date and end_date
start_year = datetime.datetime.strptime(start_date, '%Y-%m-%d').year
end_year = end_date.year
years = range(start_year, end_year + 1)

# Fetch and print historical data for each stock
for symbol in tqdm(symbols, desc="Fetching data", unit="stock"):
    symbol_data = []
    for year in years:
        try:
            start_date_year = f"{year}-01-01"
            end_date_year = f"{year}-12-31"
            stock = yf.Ticker(symbol)
            historical_data = stock.history(start=start_date_year, end=end_date_year, interval="1d")
            historical_data["Symbol"] = symbol
            symbol_data.append(historical_data)
        except Exception as e:
            print(f"Error fetching data for {symbol} in {year}: {e}")
    if symbol_data:
        all_data.append(pd.concat(symbol_data, axis=0))

# Combine all the historical data into a single DataFrame
combined_data = pd.concat(all_data, axis=0)

# Reset the index to move the date to a separate column
combined_data.reset_index(inplace=True)

# Remove timezone information and keep only the date part
combined_data['Date'] = combined_data['Date'].apply(lambda x: x.date())

# Select essential fields for stock trading
combined_data = combined_data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Adj Close']]

# Save the combined historical data to a CSV file
combined_data.to_csv("symbol_historical_data.csv", index=False)

