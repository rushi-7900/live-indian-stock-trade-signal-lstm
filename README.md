# 📈 Live Indian Stock Market Trade Signal using LSTM

A machine learning project that predicts the next trading price of selected Indian stocks using an LSTM (Long Short-Term Memory) neural network and generates educational BUY, SELL, or HOLD trade signals through a Streamlit web application.

> **Disclaimer:** This project is developed for educational and research purposes only. It does **not** provide financial or investment advice.

---

# 📌 Project Overview

This application fetches live stock market data from Yahoo Finance, trains an LSTM model, predicts the next day's closing price, and generates a trade signal based on the predicted price.

The application is built using Streamlit, TensorFlow, Scikit-learn, and the Yahoo Finance API. 

---

# 🎯 Features

- Live stock price data from Yahoo Finance
- Next-day stock price prediction using LSTM
- BUY / SELL / HOLD signal generation
- Interactive Streamlit interface
- Model performance using RMSE
- One-year stock price trend visualization

---

# 📊 Supported Stocks

- TCS
- Infosys
- Reliance Industries
- HDFC Bank
- ICICI Bank

---

# 🛠 Technologies Used

- Python
- Streamlit
- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Yahoo Finance API

---

# 🧠 Machine Learning Workflow

1. Download historical stock prices
2. Clean and preprocess the data
3. Normalize data using MinMaxScaler
4. Create LSTM sequences
5. Train the LSTM model
6. Predict the next closing price
7. Compare predicted and current prices
8. Generate BUY / SELL / HOLD signal

---

# 📂 Project Structure

```
live-indian-stock-trade-signal-lstm/

│── app.py
│── model.py
│── requirements.txt
│── README.md
│── Images/
│     ├── home-page.png
│     ├── prediction-result.png
│     └── price-chart.png
```

---

# 📸 Application Preview

## Home Screen

(Add screenshot here)

```markdown
![Home](Images/home-page.png)
```

## Prediction Result

(Add screenshot here)

```markdown
![Prediction](Images/prediction-result.png)
```

---

# 📈 Model Output

The application displays:

- Current Stock Price
- Predicted Next Price
- BUY / SELL / HOLD Signal
- RMSE
- One-Year Price Trend

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/rushi-7900/live-indian-stock-trade-signal-lstm.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📦 Requirements

Dependencies are listed in `requirements.txt`, including NumPy, Pandas, yfinance, Scikit-learn, TensorFlow, Streamlit, and Matplotlib. :contentReference[oaicite:1]{index=1}

---

# 💡 Skills Demonstrated

- Machine Learning
- Deep Learning (LSTM)
- Time Series Forecasting
- Streamlit
- Data Preprocessing
- Feature Scaling
- Model Evaluation
- Financial Data Analysis

---

# 🚀 Future Improvements

- Support more NSE/BSE stocks
- Multi-day price prediction
- Technical indicators (RSI, MACD, Bollinger Bands)
- Model comparison with GRU/XGBoost
- Deploy on Streamlit Cloud

---

# 👨‍💻 Author

**Rushikesh Sultane**

GitHub: https://github.com/rushi-7900

---

## ⭐ If you found this project useful, consider giving it a Star!
