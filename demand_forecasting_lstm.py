import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Step 1: Create Sample Supermarket Data
# -------------------------------

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Base demand with a repeating seasonal pattern
base_demand = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 10 + 100  # base around 100 units
noise = np.random.normal(0, 3, len(dates))
demand = base_demand + noise

# External factors (example: weather, offers, holidays)
df = pd.DataFrame({
    'date': dates,
    'demand': demand,
    'temperature': np.random.uniform(25, 35, len(dates)),   # daily temperature
    'rainfall': np.random.uniform(0, 10, len(dates)),       # daily rainfall in mm
    'promotion': np.random.choice([0, 1], len(dates)),      # 0 = no offer, 1 = offer
})

# Add time-based features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print("✅ Sample Data:")
print(df.head())

# -------------------------------
# Step 2: Preprocess and Prepare Data
# -------------------------------

features = ['demand', 'temperature', 'rainfall', 'promotion', 'day_of_week', 'month', 'weekend']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Function to create time sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # predicting demand
    return np.array(X), np.array(y)

seq_length = 7  # using past 7 days to predict next day
X, y = create_sequences(scaled_data, seq_length)

# Split into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------------
# Step 3: Build and Train LSTM Model
# -------------------------------

model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, len(features)), return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("\n🚀 Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# -------------------------------
# Step 4: Predictions and Evaluation
# -------------------------------

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse scale the predictions to original demand
train_unscaled = np.zeros((len(train_pred), len(features)))
train_unscaled[:, 0] = train_pred.flatten()
train_unscaled = scaler.inverse_transform(train_unscaled)[:, 0]

test_unscaled = np.zeros((len(test_pred), len(features)))
test_unscaled[:, 0] = test_pred.flatten()
test_unscaled = scaler.inverse_transform(test_unscaled)[:, 0]

# Date ranges
train_dates = df['date'][seq_length:seq_length+len(train_pred)]
test_dates = df['date'][seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)]

# -------------------------------
# Step 5: Plot Results
# -------------------------------

plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['demand'], label='Actual Demand', color='blue', alpha=0.6)
plt.plot(train_dates, train_unscaled, label='Training Prediction', linestyle='--', color='green')
plt.plot(test_dates, test_unscaled, label='Testing Prediction', linestyle='--', color='orange')
plt.title('Demand Forecasting in Supply Chain using LSTM')
plt.xlabel('Date')
plt.ylabel('Product Demand')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 6: Forecast Future Demand (Next 30 Days)
# -------------------------------

future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=30, freq='D')

last_sequence = scaled_data[-seq_length:]
future_predictions = []

for _ in range(len(future_dates)):
    next_pred = model.predict(last_sequence.reshape(1, seq_length, len(features)))
    next_pred_unscaled = np.zeros((1, len(features)))
    next_pred_unscaled[0, 0] = next_pred[0, 0]
    next_pred_unscaled = scaler.inverse_transform(next_pred_unscaled)[0, 0]
    future_predictions.append(next_pred_unscaled)
    new_row = np.zeros(len(features))
    new_row[0] = next_pred[0, 0]
    last_sequence = np.vstack((last_sequence[1:], new_row))

# Plot future forecast
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['demand'], label='Past Demand', color='blue')
plt.plot(future_dates, future_predictions, label='Forecasted Demand', color='red', linestyle='--')
plt.title('Future Demand Forecast (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Predicted Demand')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 7: Performance Metrics
# -------------------------------

train_mae = mean_absolute_error(df['demand'][seq_length:seq_length+len(train_pred)], train_unscaled)
test_mae = mean_absolute_error(df['demand'][seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)], test_unscaled)

train_rmse = np.sqrt(mean_squared_error(df['demand'][seq_length:seq_length+len(train_pred)], train_unscaled))
test_rmse = np.sqrt(mean_squared_error(df['demand'][seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)], test_unscaled))

print("\n📊 Performance Metrics:")
print(f"Training MAE: {train_mae:.2f}")
print(f"Testing MAE: {test_mae:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")

# -------------------------------
# Step 8: Display Future Forecast Table
# -------------------------------

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Demand': np.round(future_predictions, 2)
})

print("\n🔮 Future Demand Forecast:")
print(forecast_df.to_string(index=False))
