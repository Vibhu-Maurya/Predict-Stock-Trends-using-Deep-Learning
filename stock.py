import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import os
import requests

def train_model(stock="POWERGRID.NS", epochs=50):
    print(f"Starting model training pipeline for stock: {stock}...")
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2024, 11, 1)

    # Download stock data
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        df = yf.download(stock, start=start, end=end, session=session)
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        # Try fallback dataset
        fallback = f"static/{stock}_dataset.csv"
        if os.path.exists(fallback):
            print(f"Loading local fallback file: {fallback}")
            df = pd.read_csv(fallback, header=[0, 1], index_col=0, parse_dates=True)
        else:
            raise RuntimeError(f"Could not load data for training on stock: {stock}")

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # We only need the Close price column
    df_close = pd.DataFrame(df['Close'])

    # Data splitting (70% training, 30% testing)
    train_len = int(len(df_close) * 0.70)
    data_training = pd.DataFrame(df_close['Close'][0:train_len])
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    
    # Prepare training sequences (100 days lookback)
    x_train = []
    y_train = []
    
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"Training data shape: {x_train.shape}")
    
    # Model Building
    print("Building multi-layer LSTM model...")
    model = Sequential()
    
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"Starting model training for {epochs} epochs...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    
    # Save the model
    model_name = 'stock_dl_model.h5'
    model.save(model_name)
    print(f"Model successfully trained and saved to: {model_name}!")

if __name__ == '__main__':
    # Train the model (uses default POWERGRID.NS stock and 50 epochs)
    train_model(stock="POWERGRID.NS", epochs=50)
