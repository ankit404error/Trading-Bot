
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data
data = pd.read_csv('Sensex.csv', skiprows=1)

# Clean column names
data.columns = data.columns.str.strip()

# Rename columns for clarity
data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Volume']

# Drop the first row if it still contains header
data = data.drop(index=0)

# Convert date and set index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Return'] = data['Close'].pct_change()
data['Price_Change'] = data['Close'].diff()
data['Volatility'] = data['Close'].rolling(window=5).std()
data['Momentum'] = data['Close'] - data['Close'].shift(5)

# Drop NaN values
data.dropna(inplace=True)

# Define features and target
X = data[['SMA_5', 'SMA_20', 'Return', 'Price_Change', 'Volatility', 'Momentum']]
y = (data['Close'].shift(-1) > data['Close']).astype(int)

# Remove last row since target is NaN
X = X.iloc[:-1]
y = y.iloc[:-1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'nifty_model.pkl')
print("Model saved to nifty_model.pkl")
