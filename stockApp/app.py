from flask import Flask, render_template, request, flash
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.callbacks import EarlyStopping


app = Flask(__name__)

def calculate(stockname):
    stock_symbol = stockname
    start_date = '2018-01-01'
    end_date = '2023-10-31'
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data.to_csv('stock_price.csv')
    df=pd.read_csv("stock_price.csv")
    df.isnull().sum()
    df['Adj Close'].plot()
    plt.title('AAPL Stock Price Over Time')
    y=df['Adj Close']
    x=df[['Open','High','Low','Volume']]
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(x)
    feature_transform= pd.DataFrame(feature_transform,columns=['Open','High','Low','Volume'])
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    y_scaled = pd.Series(y_scaled.flatten(), name='Adj Close')
    timesplit= TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform):
            X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
            y_train, y_test = y_scaled[:len(train_index)].values.ravel(), y_scaled[len(train_index): (len(train_index)+len(test_index))].values.ravel()
    trainX =np.array(X_train)
    testX =np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    lstm = Sequential()
    lstm.add(LSTM(64, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks=[early_stopping])
    y_pred= lstm.predict(X_test)
    plt.plot(y_test, label='True Value')
    plt.plot(y_pred, label='LSTM Value')
    plt.title("Prediction by LSTM")
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.show()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sname", methods=["POST","GET"])
def show():
    calculate(str(request.form['inpName']))
    return render_template("index.html")

