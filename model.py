import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare_lstm_data(data, time_step=60):

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_lstm(X, y):

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = Sequential()

    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X.shape[1], 1)
        )
    )

    model.add(Dropout(0.2))

    model.add(LSTM(50))

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    predictions = model.predict(X_test, verbose=0)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions.flatten()
        )
    )

    return model, rmse


# -----------------------------
# NEXT DAY PREDICTION
# -----------------------------
def predict_next_price(model, data, scaler):

    last_60 = data.tail(60).values

    last_60_scaled = scaler.transform(last_60)

    last_60_scaled = last_60_scaled.reshape(1, 60, 1)

    prediction = model.predict(
        last_60_scaled,
        verbose=0
    )

    prediction = scaler.inverse_transform(prediction)

    return float(prediction[0][0])


# -----------------------------
# TRADE SIGNAL
# -----------------------------
def generate_trade_signal(
    current_price,
    predicted_price,
    threshold=0.5
):

    difference = predicted_price - current_price

    if difference > threshold:
        return "BUY 📈"

    elif difference < -threshold:
        return "SELL 📉"

    else:
        return "HOLD ⏸"