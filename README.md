# StockPulse | AI Stock Trend Predictor

StockPulse is a premium web application built with Flask and TensorFlow Keras to analyze historical stock data and predict future pricing trends using a Deep Learning **LSTM (Long Short-Term Memory) Recurrent Neural Network**. 

Featuring a modern, dark-mode glassmorphic user interface, it provides investors with a high-end dashboard to evaluate stock metrics and visualize moving averages.

---

## 🚀 Key Features

* **AI-Powered Forecasting**: Predicts future stock prices using a trained multi-layer LSTM neural network.
* **Moving Average Visualizations**: Evaluates short-term (20 & 50 Days) and long-term (100 & 200 Days) Exponential Moving Averages (EMAs).
* **Robust Fallback Engine**: Seamlessly falls back to local pre-loaded historical datasets (e.g., AAPL, POWERGRID.NS) if Yahoo Finance API encounters network rate limits.
* **Premium User Interface**: Modern responsive UI with dark aesthetics, glassmorphism cards, interactive stock selection tags, blur loaders, and data table rendering.
* **Dataset Exporting**: Export downloaded real-time historical pricing data directly to a CSV file.

---

## 🛠️ Model Architecture

The deep learning model is a Sequential network consisting of 4 LSTM layers stacked with dropout regularization to prevent overfitting:
- **LSTM Layer 1**: 50 units (sequence input) + Dropout (0.2)
- **LSTM Layer 2**: 60 units + Dropout (0.3)
- **LSTM Layer 3**: 80 units + Dropout (0.4)
- **LSTM Layer 4**: 120 units + Dropout (0.5)
- **Dense Output Layer**: 1 unit (predicted close price)

It is compiled using the **Adam Optimizer** and trained on **Mean Squared Error (MSE)** loss.

---

## 📈 Evaluation Metrics (Apple - AAPL Test Split)

- **R-squared ($R^2$) Score**: `0.9389` (explains ~93.89% of trend variance)
- **Trend/Directional Accuracy**: `79.85%` (100 - MAPE)
- **Mean Absolute Error (MAE)**: `11.64 USD`
- **Root Mean Squared Error (RMSE)**: `14.35 USD`

---

## 💻 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.12** installed (TensorFlow 2.19/Keras 3 supports Python 3.12 64-bit on Windows/macOS/Linux).

### 2. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install flask yfinance pandas numpy matplotlib scikit-learn tensorflow keras h5py requests
```

### 3. Running the Application
Launch the Flask development server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to start analyzing stock trends!

---

## 📂 Project Structure

```
├── app.py                      # Main Flask application and predictive backend
├── stock_dl_model.h5           # Pre-trained multi-layer LSTM Keras model
├── templates/
│   └── index.html              # Premium dark-mode dashboard interface
├── static/                     # Directory for generated chart plots
│   ├── AAPL_dataset.csv        # Fallback local dataset for Apple stock
│   └── POWERGRID.NS_dataset.csv # Fallback local dataset for PowerGrid stock
└── README.md                   # Project documentation
```

---

## 🤝 Contribution & License
Created by [Vibhu Maurya](https://github.com/Vibhu-Maurya). Feel free to fork, open issues, and submit pull requests. Licensed under the MIT License.
