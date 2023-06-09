import re
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
from tqdm import tqdm
from termcolor import colored
from prettytable import PrettyTable
from pandas_datareader import DataReader
from pandas_datareader._utils import RemoteDataError

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(input_seq.device), torch.zeros(1, 1, self.hidden_layer_size).to(input_seq.device))
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1), hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def get_nasdaq_symbols():
    url = 'http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
    symbols = pd.read_csv(url, sep='|')['Symbol'].dropna().astype(str).tolist()
    return symbols


def get_most_active_stocks(symbols, year):
    print(colored('\nRetrieving Most Active Stocks...\n', 'green'))
    volumes = {}
    for symbol in tqdm(symbols, desc="Downloading"):
        try:
            data = yf.download(symbol, start=f'{year}-01-01', end=f'{year}-12-31')
            if not data.empty:
                volumes[symbol] = data['Volume'].sum()
        except RemoteDataError:
            continue

    # Sort the dictionary by value in descending order and get the top 100
    sorted_volumes = dict(sorted(volumes.items(), key=lambda item: item[1], reverse=True)[:100])

    return sorted_volumes

def download_data(stock_name):
    try:
        stock = yf.Ticker(stock_name)
        hist = stock.history(period="5y")
        return hist
    except Exception as e:
        print("Error occurred during data download:", str(e))
        return None

def prepare_data(hist):
    data = hist[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    train_size = int(len(data) * 0.67)
    train, test = data[0:train_size, :], data[train_size:len(data), :]
    return train, test, scaler, data

def create_dataset(data, look_back=3):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def plot_essential_metrics(hist):
    # Define the essential financial metrics
    metrics = ["Open", "Close", "High", "Low", "Volume", "Dividends", "Stock Splits"]

    # Create a 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Essential Financial Metrics", fontsize=16)

    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics):
        hist[metric].plot(ax=axs[i // 3, i % 3], title=metric)

    # Remove the extra subplot (we have only 7 metrics and 9 subplots)
    axs[2, 1].axis("off")
    axs[2, 2].axis("off")

    # Display the plot
    plt.tight_layout()
    plt.show()
# Set the random seed for reproducible results
np.random.seed(7)

symbols = get_nasdaq_symbols()
most_active_stocks = get_most_active_stocks(symbols, 2023)

# Present most active stocks in a table
x = PrettyTable()
x.field_names = ["Stock", "Volume"]
for stock, volume in most_active_stocks.items():
    x.add_row([stock, volume])
print(x)

# Let's consider the most active stock
stock_name = list(most_active_stocks.keys())[0]

hist = download_data(stock_name)
train, test, scaler, data = prepare_data(hist)
trainX, trainY = create_dataset(train, look_back=3)
testX, testY = create_dataset(test, look_back=3)

model = LSTM(3, 100, 1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Model training loop with a progress bar
print(colored('\nTraining Model...\n', 'green'))
for epoch in tqdm(range(100), desc="Training"):
    model.zero_grad()
    output = model(Variable(torch.from_numpy(trainX).type(torch.Tensor)))
    loss = loss_function(output, Variable(torch.from_numpy(trainY).type(torch.Tensor)))
    loss.backward()
    optimizer.step()

plot_essential_metrics(hist)
