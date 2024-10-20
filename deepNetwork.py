import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Transformation des données en tenseurs et normalisation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Chargement du dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Fonction pour créer un DataLoader avec un batch size variable
def get_data_loaders(batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class DeepNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_hidden_layers):
        super(DeepNeuralNet, self).__init__()
        self.layers = nn.ModuleList()  # Liste pour les couches cachées

        # Limiter le nombre de couches cachées au nombre d'éléments dans hidden_sizes
        num_hidden_layers = min(num_hidden_layers, len(hidden_sizes))

        # Première couche
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Couches cachées supplémentaires
        for i in range(1, num_hidden_layers):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

        # Couche de sortie
        self.output_layer = nn.Linear(hidden_sizes[num_hidden_layers-1], output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Aplatir les images
        for layer in self.layers:
            x = torch.relu(layer(x))  # Passer à travers chaque couche cachée avec ReLU
        x = self.output_layer(x)  # Pas de ReLU pour la couche de sortie
        return x

# Fonction d'entraînement
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Mettre le modèle en mode entraînement
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zéro du gradient
            outputs = model(images)  # Propagation avant
            loss = criterion(outputs, labels)  # Calcul de la perte
            loss.backward()  # Rétropropagation
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

# Fonction de test
def test(model, test_loader):
    model.eval()  # Mode évaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total  # Retourne la précision

# Liste des hyperparamètres à explorer
learning_rates = [0.001, 0.01, 0.1, 0.3]
hidden_sizes_list = [[128, 64], [256, 128], [64, 32]]
batch_sizes = [32, 64, 128]
num_hidden_layers_options = [2, 3]
epochs = 10

best_accuracy = 0
best_params = {}

# Boucle pour tester toutes les combinaisons d'hyperparamètres
for learning_rate in learning_rates:
    for hidden_sizes in hidden_sizes_list:
        for num_hidden_layers in num_hidden_layers_options:
            for batch_size in batch_sizes:
                print(f"\nTesting configuration: LR={learning_rate}, Hidden sizes={hidden_sizes}, "
                      f"Num hidden layers={num_hidden_layers}, Batch size={batch_size}")

                # Charger les données avec le batch size actuel
                train_loader, test_loader = get_data_loaders(batch_size)

                # Initialiser le modèle avec les hyperparamètres actuels
                model = DeepNeuralNet(input_size=28*28, hidden_sizes=hidden_sizes, output_size=10, num_hidden_layers=num_hidden_layers)

                # Définir la fonction de perte et l'optimiseur
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                # Entraîner le modèle
                train(model, train_loader, criterion, optimizer, epochs=epochs)

                # Tester le modèle
                accuracy = test(model, test_loader)
                print(f"Accuracy: {accuracy}%")

                # Mémoriser les meilleurs hyperparamètres
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'learning_rate': learning_rate,
                        'hidden_sizes': hidden_sizes,
                        'num_hidden_layers': num_hidden_layers,
                        'batch_size': batch_size
                    }

# Afficher la meilleure configuration
print(f"\nBest Accuracy: {best_accuracy}%")
print(f"Best Parameters: {best_params}")
