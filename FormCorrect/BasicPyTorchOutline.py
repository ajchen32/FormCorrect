import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer (2 features) to hidden layer (4 neurons)
        self.fc2 = nn.Linear(4, 1)  # Hidden layer to output layer (1 neuron)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.sigmoid(self.fc2(x))  # Apply Sigmoid activation
        return x

# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Generate random training data
X = torch.rand(10, 2)  # 10 samples, 2 features each
y = torch.randint(0, 2, (10, 1)).float()  # Binary target labels

# Training loop
for epoch in range(100):
    optimizer.zero_grad()  # Zero gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')



