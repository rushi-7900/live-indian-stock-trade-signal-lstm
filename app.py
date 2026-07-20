import streamlit as st
import yfinance as yf

from model import (
    prepare_lstm_data,
    train_lstm,
    predict_next_price,
    generate_trade_signal
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Live Indian Stock Trade Signal",
    layout="centered"
)

st.title("📊 Live Indian Stock Market – Next Trade Signal")
st.caption("Educational Decision Support System (Not Financial Advice)")

# -----------------------------
# STOCK SELECTION
# -----------------------------
stock = st.selectbox(
    "Select Indian Stock (NSE)",
    [
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS"
    ]
)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("Get Next Trade Signal"):

    with st.spinner("Fetching live market data & training model..."):

        try:
            # -----------------------------
            # DOWNLOAD DATA
            # -----------------------------
            data = yf.download(
                stock,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if data.empty:
                st.error("No data received from Yahoo Finance.")
                st.stop()

            # -----------------------------
            # FIX MULTIINDEX
            # -----------------------------
            if hasattr(data.columns, "levels"):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()
            data = data.dropna()

            # -----------------------------
            # CHECK REQUIRED COLUMN
            # -----------------------------
            if "Close" not in data.columns:
                st.error(f"'Close' column not found.\nAvailable columns:\n{list(data.columns)}")
                st.stop()

            # -----------------------------
            # MODEL DATA
            # -----------------------------
            close_data = data[["Close"]].copy()
            close_data_plot = data[["Date", "Close"]].copy()

            # -----------------------------
            # PREPARE DATA
            # -----------------------------
            X, y, scaler = prepare_lstm_data(close_data)

            # -----------------------------
            # TRAIN MODEL
            # -----------------------------
            model, rmse = train_lstm(X, y)

            # -----------------------------
            # PREDICT
            # -----------------------------
            predicted_price = predict_next_price(
                model,
                close_data,
                scaler
            )

            # -----------------------------
            # CURRENT PRICE
            # -----------------------------
            current_price = float(close_data["Close"].iloc[-1])

            # -----------------------------
            # SIGNAL
            # -----------------------------
            signal = generate_trade_signal(
                current_price,
                predicted_price
            )

            # -----------------------------
            # DISPLAY
            # -----------------------------
            st.success("Analysis Completed ✅")

            st.subheader("📌 Market Prices")

            st.metric(
                "Current Price",
                f"₹{current_price:.2f}"
            )

            st.metric(
                "Predicted Next Price",
                f"₹{predicted_price:.2f}"
            )

            st.subheader("🧠 Suggested Next Action")

            if "BUY" in signal:
                st.success(signal)

            elif "SELL" in signal:
                st.error(signal)

            else:
                st.warning(signal)

            st.subheader("📉 Model Performance")

            st.write(f"**RMSE:** {rmse:.4f}")

            st.subheader("📈 Last 1 Year Price Trend")

            st.line_chart(
                close_data_plot.set_index("Date")
            )

            st.warning(
                "⚠ This application is for educational purposes only and should not be considered financial advice."
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)