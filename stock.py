import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import os
import requests
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

def train_model(stock="POWERGRID.NS", epochs=50):
    print(f"Starting model training pipeline for stock: {stock}...")
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2024, 11, 1)

    df = pd.DataFrame()
    
    # Try downloading first
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        df = yf.download(stock, start=start, end=end, session=session)
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        df = pd.DataFrame()

    # If download is empty, check for local fallback files
    if df.empty or len(df) < 150:
        print(f"Download returned no data. Searching for local fallback dataset for {stock}...")
        fallback_paths = [
            f"static/{stock}_dataset.csv",
            f"static/{stock.upper()}_dataset.csv",
            f"{stock.lower()}_dataset.csv",
            "powergrid.csv" if "POWERGRID" in stock.upper() else None
        ]
        # Filter existing paths
        fallback_paths = [p for p in fallback_paths if p and os.path.exists(p)]
        
        if fallback_paths:
            csv_path = fallback_paths[0]
            print(f"Loading local fallback file: {csv_path}")
            try:
                # Try reading with MultiIndex headers first
                df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
                if 'Close' not in df.columns.get_level_values(0):
                    # Try single header
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except Exception as read_err:
                print(f"Error reading local CSV {csv_path}: {read_err}")
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                except Exception:
                    pass
                    
    # Flatten columns if they are MultiIndex
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 150:
        raise RuntimeError(f"Could not load sufficient data for training on stock: {stock}")

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
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    
    # Save the model
    model_name = 'stock_dl_model.h5'
    model.save(model_name)
    print(f"Model successfully trained and saved to: {model_name}!")
    
    # Generate and save model training loss graph
    print("Generating training loss graph...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], 'r', label='Training Loss (MSE)', linewidth=2)
    ax.set_title(f"Model Training Loss over Epochs ({stock})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    os.makedirs('static', exist_ok=True)
    loss_chart_path = "static/training_loss.png"
    fig.savefig(loss_chart_path, bbox_inches='tight', facecolor='#1f2328', edgecolor='none')
    plt.close(fig)
    print(f"Training loss graph saved to: {loss_chart_path}")

if __name__ == '__main__':
    # Train the model (uses default POWERGRID.NS stock and 50 epochs)
    train_model(stock="POWERGRID.NS", epochs=50)
