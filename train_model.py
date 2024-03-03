import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool
from data_generator import DataGenerator
from model import NeuralNetwork

MAX_NOBS = 10000
NUM_SIMULATIONS = 100

BATCH_SIZE = 64
NUM_EPOCHS = 1000

# Instantiate the neural network
model = NeuralNetwork()
data_generator = DataGenerator(MAX_NOBS, NUM_SIMULATIONS)

# # Define training data (example)
# X_train = torch.randn(100, 4)  # 100 samples, 4 features
# y_train = torch.randn(100, 1)  # 100 samples, 1 output

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network
for epoch in range(NUM_EPOCHS):
    
    # Generate data (this needs to be pararelised)
    X_train = []
    y_train = []

    with Pool() as pool:
        pool_output = pool.starmap(data_generator.generate, [() for _ in range(BATCH_SIZE)])
    X_train = [x for x, y in pool_output]
    y_train = [y for x, y in pool_output]

    X_train = torch.Tensor(X_train)
    
    y_train = torch.Tensor(y_train)
    y_train.resize_(BATCH_SIZE, 1)

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    
    if not torch.isnan(loss):
      loss.backward()
      optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')