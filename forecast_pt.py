import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


'''
csv file is diabetes.csv:

Predictor columns are:
Pregnancies	
Glucose	
BloodPressure	
SkinThickness	
Insulin	
BMI	
Age	

Target column is Outcome (0 or 1)
Outcome
'''

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Split the data into features and target
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the neural network model
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model, loss function, and optimizer
model = DiabetesModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.round()
    accuracy = (predictions.eq(y_test).sum() / float(y_test.shape[0])).item()
    print(f'Accuracy: {accuracy:.4f}')

# Save the model and scaler
torch.save(model.state_dict(), 'diabetes_model.pth')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Function to load the model and scaler, and make predictions
def predict_diabetes(input_data):
    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Convert input data to DataFrame to match scaler's expected input format
    input_data = pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])
    
    # Standardize the input data
    input_data = scaler.transform(input_data)
    input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Load the model
    model = DiabetesModel()
    model.load_state_dict(torch.load('diabetes_model.pth'))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_data)
        prediction = prediction.round().item()
    
    return prediction

# Example usage
input_data = [2, 120, 70, 30, 100, 25.0, 35]  # Example input data
print(f'Prediction: {predict_diabetes(input_data)}')