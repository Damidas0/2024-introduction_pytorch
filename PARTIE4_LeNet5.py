import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self, hidden_neurons = [120,84] ):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 canal (grayscale), 6 filtres, noyau 5x5
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling avec noyau 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 16 filtres, noyau 5x5
        self.fc1 = nn.Linear(256, hidden_neurons[0])  # 16 canaux de 4x4, 120 unités
        self.fc2 = nn.Linear(hidden_neurons[0], hidden_neurons[1])
        self.fc3 = nn.Linear(hidden_neurons[1], 10)  # 10 classes de sortie (MNIST)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = x.view(-1, 16 * 4 * 4)  # Mise à plat
        x = torch.relu(self.fc1(x))  # Fully connected 1 + ReLU
        x = torch.relu(self.fc2(x))  # Fully connected 2 + ReLU
        x = self.fc3(x)  # Sortie
        return x


def train_model(batch_size, learning_rate, nb_hidden_neurons) :
     
    #Données MNIST
    try : 
        del modele 
    except : 
        pass
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalisation des images MNIST (0.5, 0.5)
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    #Initialisation du réseau
    modele = LeNet5(nb_hidden_neurons)

    #Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()  # Perte de classification multi-classes
    optimizer = optim.Adam(modele.parameters(), lr=learning_rate)  # Optimiseur Adam

    
    epochs=10
    train_losses = []
    accuracy_tab = []
    for epoch in range(epochs):
        running_loss = 0.0
        modele.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()  # Réinitialisation des gradients
            outputs = modele(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

        # Calcul de la perte moyenne d'entraînement
        train_losses.append(running_loss / len(trainloader))

        # Validation après chaque époque
        correct, total = 0, 0
        modele.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = modele(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calcul de la perte et de la précision
        accuracy = correct / total * 100
        accuracy_tab.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.3f}, Accuracy: {accuracy:.2f}%")

    return train_losses, accuracy_tab

# --------------OLD---------------
def plot_learning_curves(train_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()

def plot_accuracry_curves(accuracy_tab) :
    plt.plot(accuracy_tab, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Évolution de la précision')
    plt.show()
#------------------------------------




plt.figure(figsize=(6, 6))
c = 0
cmap=plt.get_cmap('terrain')
for i in [[32,32], [32,64], [64,64], [64,120], [120,84], [128,128], [128,64]] : 
    start_time = time.time()

    train_losses, accuracy_tab = train_model(64, 0.001,i)
    elapsed_time = time.time() - start_time
    
    plt.plot(train_losses, label=f'Loss pour les neuronnes = {str(i)} in {elapsed_time}', color=cmap(c))
    c = c+0.15


# Calculer le temps écoulé
plt.title('Evolution de la loss en fonction de la taille des batchs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Affichage des graphes
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()