import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Step 1: Load Dataset (NEW CHANGE)
# -------------------------------

df = pd.read_csv("supermarket_demand_data.csv")
df['date'] = pd.to_datetime(df['date'])

print("✅ Loaded Dataset:")
print(df.head())

# -------------------------------
# Step 2: Preprocess Data
# -------------------------------

features = ['demand', 'temperature', 'rainfall', 'promotion', 'day_of_week', 'month', 'weekend']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # demand column
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_sequences(scaled_data, seq_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------------
# Step 3: Build LSTM Model
# -------------------------------

model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, len(features)), return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("\n🚀 Training the model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# -------------------------------
# Step 4: Predictions
# -------------------------------

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse scaling
train_unscaled = np.zeros((len(train_pred), len(features)))
train_unscaled[:, 0] = train_pred.flatten()
train_unscaled = scaler.inverse_transform(train_unscaled)[:, 0]

test_unscaled = np.zeros((len(test_pred), len(features)))
test_unscaled[:, 0] = test_pred.flatten()
test_unscaled = scaler.inverse_transform(test_unscaled)[:, 0]

# Dates
train_dates = df['date'][seq_length:seq_length+len(train_pred)]
test_dates = df['date'][seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)]

# -------------------------------
# Step 5: Plot Results
# -------------------------------

plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['demand'], label='Actual Demand')
plt.plot(train_dates, train_unscaled, '--', label='Training Prediction')
plt.plot(test_dates, test_unscaled, '--', label='Testing Prediction')

plt.title('Demand Forecasting using LSTM')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 6: Future Prediction (30 days)
# -------------------------------

future_dates = pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=30)

last_sequence = scaled_data[-seq_length:]
future_predictions = []

for _ in range(len(future_dates)):
    next_pred = model.predict(last_sequence.reshape(1, seq_length, len(features)))
    
    temp = np.zeros((1, len(features)))
    temp[0, 0] = next_pred[0, 0]
    value = scaler.inverse_transform(temp)[0, 0]
    
    future_predictions.append(value)
    
    new_row = np.zeros(len(features))
    new_row[0] = next_pred[0, 0]
    last_sequence = np.vstack((last_sequence[1:], new_row))

# Plot future
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['demand'], label='Past Demand')
plt.plot(future_dates, future_predictions, '--', label='Future Forecast')

plt.title('30-Day Future Demand Forecast')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 7: Metrics
# -------------------------------

train_mae = mean_absolute_error(df['demand'][seq_length:seq_length+len(train_pred)], train_unscaled)
test_mae = mean_absolute_error(df['demand'][seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)], test_unscaled)

train_rmse = np.sqrt(mean_squared_error(df['demand'][seq_length:seq_length+len(train_pred)], train_unscaled))
test_rmse = np.sqrt(mean_squared_error(df['demand'][seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)], test_unscaled))

print("\n📊 Performance:")
print("Train MAE:", round(train_mae, 2))
print("Test MAE:", round(test_mae, 2))
print("Train RMSE:", round(train_rmse, 2))
print("Test RMSE:", round(test_rmse, 2))
