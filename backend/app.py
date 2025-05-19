
from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = joblib.load('nifty_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load historical data once
try:
    historical_data = pd.read_csv('Sensex.csv', skiprows=1)
    historical_data.columns = historical_data.columns.str.strip()
    historical_data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Volume']
    historical_data = historical_data.drop(index=0)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data.set_index('Date', inplace=True)
except Exception as e:
    print(f"Error loading historical data: {e}")
    historical_data = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if historical_data is None:
        return jsonify({'error': 'Historical data not loaded'}), 500

    data = request.get_json()

    # Validate required fields
    required_fields = ['current', 'changePercent']
    if not all(field in data for field in required_fields):
        return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400

    try:
        # Append latest market data to historical data
        latest_date = pd.Timestamp.now()
        latest_data = {
            'Price': data['current'],  # current price
            'Close': data['current'],  # assuming current price as close
            'High': data['current'],   # approximate
            'Low': data['current'],    # approximate
            'Volume': 0                # unknown
        }
        temp_data = historical_data.copy()
        temp_data.loc[latest_date] = latest_data

        # Compute previous close as last working day's close from historical data
        previous_close = historical_data['Close'].iloc[-1]

        # Feature engineering
        temp_data['SMA_5'] = temp_data['Close'].rolling(window=5).mean()
        temp_data['SMA_20'] = temp_data['Close'].rolling(window=20).mean()
        temp_data['Return'] = temp_data['Close'].pct_change()
        temp_data['Price_Change'] = temp_data['Close'].diff()
        temp_data['Volatility'] = temp_data['Close'].rolling(window=5).std()
        temp_data['Momentum'] = temp_data['Close'] - temp_data['Close'].shift(5)

        # Drop NaN values
        temp_data.dropna(inplace=True)

        # Use the latest row for prediction
        latest_features = temp_data.iloc[-1][['SMA_5', 'SMA_20', 'Return', 'Price_Change', 'Volatility', 'Momentum']]

        # Convert to DataFrame with column names to avoid sklearn warning
        latest_features_df = pd.DataFrame([latest_features.values], columns=latest_features.index)

        print("Latest features for prediction:", latest_features_df)
        proba = model.predict_proba(latest_features_df)
        print("Prediction probabilities:", proba)
        prediction = model.predict(latest_features_df)[0]
        print("Predicted class:", prediction)
        proba_list = proba[0].tolist()
        # Calculate points change as difference between current and previous close
        points_change = data['current'] - previous_close
        return jsonify({'prediction': float(prediction), 'probabilities': proba_list, 'points_change': points_change})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Exception during prediction:\n", tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500

if __name__ == '__main__':
    app.run(debug=True)
