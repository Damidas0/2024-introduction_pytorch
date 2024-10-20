import torch.nn as nn
import torch

#Nombre neuronnes couche cachée 
class ShallowNet(nn.Module):
    def __init__(self, nb_neurons_hidden):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(784, nb_neurons_hidden)  # Hidden layer with `nb_neurons_hidden` units
        self.relu = nn.ReLU()                        # ReLU activation function
        self.fc2 = nn.Linear(nb_neurons_hidden, 10)  # Output layer with 10 units (for 10 classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (batch_size, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)            # No softmax or log-softmax here
        return x                   # Return raw logits

