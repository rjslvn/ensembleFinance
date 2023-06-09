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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(input_seq.device),
                       torch.zeros(1,1,self.hidden_layer_size).to(input_seq.device))
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1), hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def download_data(stock_name):
    stock = yf.Ticker(stock_name)
    hist = stock.history(period="5y")
    return hist

def prepare_data(hist):
    data = hist[['Close']]
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)
    train_size = int(len(data) * 0.67)
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    return train, test, scaler, data


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def main():
    stock_name = input("Enter the stock symbol (e.g., TSLA): ")
    hist = download_data(stock_name)

    # Weekly resampling
    weekly_hist = hist.resample('W').mean()

    train, test, scaler, data = prepare_data(hist)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = Variable(torch.Tensor(trainX))
    trainY = Variable(torch.Tensor(trainY))
    testX = Variable(torch.Tensor(testX))
    testY = Variable(torch.Tensor(testY))

    # Model parameters
    input_size = 1
    hidden_layer_size = 100
    output_size = 1
    model = LSTM(input_size, hidden_layer_size, output_size)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 500

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(trainX)
        loss = loss_function(out, trainY)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f'Epoch: {epoch} Loss: {loss.item()}')

    # Testing the model
    model.eval()
    predict = model(testX)

    # Plot closing price (Weekly)
    weekly_hist.plot(y='Close')
    plt.title('Weekly Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()

    # Predicting for next 26 weeks (approx. rest of 2023)
    future_weeks = 52
    forecast = []

    current_week = testX[-1:].tolist()
    for i in range(future_weeks):
        with torch.no_grad():
            model.eval()
            current_week_var = Variable(torch.Tensor(current_week))
            predict = model(current_week_var)
            current_week.append(predict.item())
            current_week.pop(0)
            forecast.append(predict.item())

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))

    # Create a list for holding predicted prices
    testPredictPlot = np.empty_like(scaler.inverse_transform(data))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainX)+(look_back*2)+1:len(data)-1, :] = scaler.inverse_transform(predict.detach().numpy().reshape(-1,1))
  
    # Plotting the actual price and predicted price on the same plot
    plt.figure(figsize=(12,6))
    plt.plot(scaler.inverse_transform(data), label='Actual Price')
    plt.plot(testPredictPlot, label='Predicted Price')

    # Creating the forecast for the next 26 weeks
    forecastPlot = np.empty((len(data)+future_weeks,1))
    forecastPlot[:, :] = np.nan
    forecastPlot[-future_weeks:] = forecast

    plt.plot(forecastPlot, label='Forecast')
    plt.title("Weekly Closing Prices & Forecast for next " + str(future_weeks) + " Weeks")
    plt.xlabel('Weeks')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
