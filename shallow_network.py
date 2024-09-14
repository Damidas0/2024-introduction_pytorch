import torch.nn as nn
import torch

#Nombre neuronnes couche cachée 


class ShallowNet(nn.Module):
    def __init__(self, nb_neurons_hidden):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(784, nb_neurons_hidden)  # 128 neurones dans la couche cachée
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nb_neurons_hidden, 10)   # 10 classes en sortie

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
