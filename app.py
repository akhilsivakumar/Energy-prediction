from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the CSV file into a DataFrame
df = pd.read_csv('events.csv')

# Preprocessing and Feature Extraction
df['Start time UTC'] = pd.to_datetime(df['Start time UTC'])
df['Month'] = df['Start time UTC'].dt.month
df['Year'] = df['Start time UTC'].dt.year
df['Electricity consumption in Finland'] = pd.to_numeric(df['Electricity consumption in Finland'], errors='coerce')

# Group by month and year, and aggregate electricity consumption
monthly_data = df.groupby(['Year', 'Month'])['Electricity consumption in Finland'].sum().reset_index()

# LSTM Model
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data['Electricity consumption in Finland'].values.reshape(-1, 1))

# Function to create dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Reshaping the data
time_step = 12
X, Y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X, Y, epochs=100, batch_size=64, verbose=0)

# Predict function
def predict_electricity_consumption(month, year):
    input_data = monthly_data[(monthly_data['Month'] == month) & (monthly_data['Year'] == year)]
    input_data = input_data['Electricity consumption in Finland'].values[-time_step:]
    input_data = scaler.transform(input_data.reshape(-1, 1))
    X_test = []
    X_test.append(input_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_consumption = model.predict(X_test)
    predicted_consumption = scaler.inverse_transform(predicted_consumption)
    return predicted_consumption[0][0]

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    month = int(request.json['month'])
    year = int(request.json['year'])
    prediction = predict_electricity_consumption(month, year)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
