# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import time
from matplotlib import pyplot as plt, transforms
import gzip, numpy, torch, matplotlib
import torchvision
from PARTIE2_shallow_network import *
from torch.utils.data import random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms



def train_return(batch_size = 5, nb_epochs = 3, eta = 0.001, hidden_neurons = 64):
	"""Entraine le modèle avec les paramètres donné 

	Args:
		batch_size (int, optional): Taille des jeux d'entrainements. Defaults to 5.
		nb_epochs (int, optional): Nombre de période d'entrainements. Defaults to 3.
		eta (float, optional): Taux d'apprentissage. Defaults to 0.001.
		hidden_neurons (int, optional): Nombre de neurones dans la couche cachées. Defaults to 64.

	Returns:
		(tab, float): (Tableau de la précision, temps d'entrainement) 
	"""

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Chargement du dataset MNIST
	train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	try:
		del(model)  
	except:
		pass

	model = ShallowNet(hidden_neurons)

	#Définiion de la fonction Loss
	loss_func = torch.nn.CrossEntropyLoss()

	optim = torch.optim.SGD(model.parameters(), lr=eta)

	test_accuracies = []
	start_time = time.time()

	model.train()
	for n in range(nb_epochs):
		for x, t in train_loader:
			y = model(x)  

			t = t.long()

			
			loss = loss_func(y, t)
			loss.backward()  
			optim.step()  # Mise à jours des poids 
			optim.zero_grad()  
			
		model.eval()
		correct = 0
		total = 0

		with torch.no_grad():
			for images, labels in test_loader:
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1) 
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		accuracy = 100 * correct / total
		print(f'Test Accuracy: {accuracy:.2f}%')
		test_accuracies.append(accuracy)

	elapsed_time = time.time() - start_time

	return test_accuracies, elapsed_time



    
if __name__ == '__main__':
	nb_epochs = 10
	batch_size = 5
	eta = 0.001
	nb_neurons = 128
	#Pour varier la couleur des plots
	cmap=plt.get_cmap('terrain')
	
	plt.figure(figsize=(6, 6))
	c = 0
 
	for i in [4,8,16,32,64,128] : #liste de paramètre qu'on veut tester
		print(i)
		test_accuracies, elapsed_time = train_return(i, nb_epochs, eta, nb_neurons)
		# Visualisation des résultats

		# Graphe de l'accuracy
		plt.plot(range(1, nb_epochs+1), test_accuracies, label=f'Taille training : {i} a duré {elapsed_time:.2f}s', color=cmap(c))
  
		c = c+0.15


	# Affichage des hyperparamètres choisis
	plt.title('Evolution de la précision en fonction de la taille des jeux d\'entrainements')
	plt.xlabel('Epochs')
	plt.ylabel('Précision')
	plt.legend()
	# Affichage des graphes
	plt.tight_layout(rect=[0, 0, 1, 1])
	plt.show()