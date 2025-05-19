import pandas as pd

# Load the data, skipping the first row and setting the correct header
data = pd.read_csv('Sensex.csv', skiprows=1)

# Display the first few rows and the columns
print(data.head())
print(data.columns)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Rename the columns for clarity (adjusting to match the number of columns)
data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Volume']

# Drop the first row if it still contains the header
data = data.drop(index=0)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Feature Engineering: Create features like moving averages, etc.
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Return'] = data['Close'].pct_change()

# Drop NaN values
data.dropna(inplace=True)

# Define features and target
X = data[['SMA_5', 'SMA_20', 'Return']]
y = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price goes up, else 0

# Display the features and target
print(X.head())
print(y.head())