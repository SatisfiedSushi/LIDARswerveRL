import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import pandas as pd

# Load and preprocess data
data = pd.read_csv('HistoricalData_Apple.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Date_days'] = (data['Date'] - data['Date'].min()).dt.days
for column in ['Close/Last', 'Open', 'High', 'Low']:
    data[column] = data[column].str.replace('$', '').astype(float)

# Use only 'Date_days' as features
features = data[['Date_days']].values  # Simplified to use only date information
target = data['Close/Last'].values


# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = X_train.shape[1]

class SimpleNN(nn.Module):
    """A simple neural network with one hidden layer, mimicking a decision tree."""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ForestRegressor:
    """An ensemble of simple neural networks to mimic a forest regressor."""

    def __init__(self, n_estimators, input_size, hidden_size, output_size, device):
        self.n_estimators = n_estimators
        self.models = [SimpleNN(input_size, hidden_size, output_size).to(device) for _ in range(n_estimators)]
        self.device = device

    def fit(self, train_loader, epochs, lr=0.01):
        for model in self.models:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            for epoch in tqdm(range(epochs), desc='Training Ensemble', unit='epoch'):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

    def predict(self, X):
        total_predictions = torch.zeros(len(X), 1, device=self.device)

        with torch.no_grad():
            for model in self.models:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                predictions = model(X_tensor)
                total_predictions += predictions

        return total_predictions / self.n_estimators


# Assuming X_train, X_test, y_train, y_test are already prepared and standardized
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
forest = ForestRegressor(n_estimators=10, input_size=input_size, hidden_size=10, output_size=1, device=device)

# Training
forest.fit(train_loader, epochs=1000, lr=0.001)

# Predicting
predictions = forest.predict(X_test).cpu().numpy().flatten()
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# Function to convert input date to 'Date_days'
def date_to_days(date_str, reference_date):
    date = pd.to_datetime(date_str)
    return (date - reference_date).days


# Prediction loop
reference_date = data['Date'].min()  # Reference date to calculate 'Date_days'
while True:
    print("Enter the date in Month/Day/Year format (e.g., 01/31/2022):")
    date_str = input()
    date_days = date_to_days(date_str, reference_date)

    # Scaling the 'Date_days' feature before prediction
    date_days_scaled = scaler.transform([[date_days]])

    # Predicting the 'Close/Last' price
    prediction = forest.predict(date_days_scaled)
    print(f'Predicted Close/Last price for {date_str}: ${prediction.item():.2f}')

    print("Do you want to predict another price? (yes/no)")
    if input().lower() == 'no':
        break

