import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(4, 10)   # 4 input neurons, 10 neurons in the first hidden layer
        self.hidden_layer1 = nn.Linear(10, 8)  # 10 neurons in the first hidden layer, 8 neurons in the second hidden layer
        self.hidden_layer2 = nn.Linear(8, 6)  # 8 neurons in the second hidden layer, 6 neurons in the third hidden layer
        self.output_layer = nn.Linear(6, 1)  # 6 neurons in the third hidden layer, 1 output neuron

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x