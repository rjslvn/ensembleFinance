import pandas as pd
import numpy as np
from prophet import Prophet
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import pandas as pd
import datetime
from tqdm import tqdm
import XGBoost 

# Get the list of symbols
end_date = datetime.date.today()
symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH', 'BAC', 'HD', 'DIS', 'VZ', 'NFLX', 'ADBE', 'PYPL', 'CRM', 'KO', 'INTC', 'CSCO', 'PFE', 'CMCSA', 'XOM', 'T', 'COST', 'PEP', 'BA', 'NVDA', 'ABT', 'CVX', 'TMO', 'MRK', 'ABBV', 'WMT', 'MCD', 'C', 'NKE', 'ACN', 'UNP', 'HON', 'ORCL', 'IBM', 'NEE', 'LLY', 'TXN', 'MMM', 'AMGN', 'DHR', 'AVGO', 'COP', 'LMT', 'SPG', 'AXP', 'MDT', 'NVS', 'LIN', 'CAT', 'SBUX', 'ADP', 'SCHW', 'GS', 'DE', 'BLK', 'HD', 'BMY', 'CI', 'BDX', 'RTX', 'D', 'DUK', 'NEE', 'WM', 'BDX', 'PLD', 'PNC', 'FIS', 'LOW', 'BDX', 'CNC', 'COF', 'GE', 'DOW', 'BK', 'CCL', 'CMA', 'CB', 'APD', 'SYY', 'SRE', 'PGR', 'PEG', 'GLW', 'DD', 'BAX', 'BLL', 'F', 'WBA', 'MET', 'MMC', 'CBRE', 'VLO', 'MMC', 'CMI', 'HON', 'AON', 'BLK', 'CB', 'CARR', 'HIG', 'TSN', 'AEP', 'AIG', 'LIN', 'WFC', 'KR', 'DTE', 'COF', 'PCAR', 'CTAS', 'MCK', 'HCA', 'ZTS', 'ZBH', 'ZION', 'ZBRA']
start_date = '2010-01-01'

all_data = []

all_data = []
start_year = datetime.datetime.strptime(start_date, '%Y-%m-%d').year
end_year = end_date.year
years = range(start_year, end_year + 1)

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

combined_data = pd.concat(all_data, axis=0)
combined_data.reset_index(inplace=True)
combined_data['Date'] = combined_data['Date'].apply(lambda x: x.date())
combined_data = combined_data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Adj Close']]
combined_data.to_csv("symbol_historical_data.csv", index=False)


# Set the number of threads
num_threads = os.cpu_count()

def train_prophet_model(data, symbol):
    df = data[data['Symbol'] == symbol][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    model = Prophet(daily_seasonality=True)

    # Add extra regressors
    model.add_regressor('Open')
    model.add_regressor('High')
    model.add_regressor('Low')
    model.add_regressor('Volume')

    model.fit(df)
    return model

def train_gbm_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

def train_xgboost_model(X, y):
    model = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
    model.fit(X, y)
    return model

def train_arima_model(data):
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit(disp=0)
    return model_fit

def train_sarima_model(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    return model_fit

def ensemble_predictions(models, data):
    predictions = []
    for model in models:
        prediction = model.predict(data)
        predictions.append(prediction)
    return np.mean(predictions, axis=0)

def train_models(data, symbol):
    X = data[data['Symbol'] == symbol][['Open', 'High', 'Low', 'Volume']].values
    y = data[data['Symbol'] == symbol]['Close'].values

    gbm_model = train_gbm_model(X, y)
    xgboost_model = train_xgboost_model(X, y)
    arima_model = train_arima_model(data['Close'])
    sarima_model = train_sarima_model(data['Close'])
    prophet_model = train_prophet_model(data, symbol)

    return gbm_model, xgboost_model, arima_model, sarima_model, prophet_model

def main():
    data = pd.read_csv('dataset.csv', parse_dates=['Date'], index_col='Date')
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].apply(lambda x: x.replace(tzinfo=None))  # Modify this line to remove timezone info
    data.drop(['Dividends', 'Stock Splits', 'Adj Close', 'Capital Gains'], axis=1, inplace=True)

    symbols = data['Symbol'].unique()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(train_models, data, symbol): symbol for symbol in symbols}

        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            gbm_model, xgboost_model, arima_model, sarima_model, prophet_model = future.result()

            # Save models
            joblib.dump(gbm_model, f'models/{symbol}_gbm_model.pkl')
            joblib.dump(xgboost_model, f'models/{symbol}_xgboost_model.pkl')
            joblib.dump(arima_model, f'models/{symbol}_arima_model.pkl')
            joblib.dump(sarima_model, f'models/{symbol}_sarima_model.pkl')
            prophet_model.save(f'models/{symbol}_prophet_model.pkl')

            print(f'Models trained and saved for {symbol}')

if __name__ == "__main__":
    main()