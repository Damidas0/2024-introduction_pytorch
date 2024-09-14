# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

from matplotlib import pyplot as plt
import gzip, numpy, torch, matplotlib
from shallow_network import *

def train_return(batch_size = 5, nb_epochs = 3, eta = 0.00001, hidden_neurons = 64) :
	"""Entraine le réseau de neuronnes et renvoi l'évolution de l'accuracy au cours des epochs en fonction des paramètres

	Args:
		batch_size (int, optional): Taille des jeu d'entrainements.
		nb_epochs (int, optional): Nombres d'époques (d'itérations)..
		eta (float, optional): Taux d'apprentissage. 
		hidden_neurons (int, optional): Nombre de neuronne de la couche cachée.
	"""
 
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	# on initialise le modèle et ses poids
	# model = torch.nn.Linear(data_train.shape[1],label_train.shape[1])
	try :
		del(model) 
	except : 
		pass
     
    
	model = ShallowNet(hidden_neurons)
# 	torch.nn.init.uniform_(model.weight,-0.001,0.001)
	
 
	# on initiliase l'optimiseur
	#Cgt de sum à mean
	loss_func = torch.nn.MSELoss(reduction='mean')
	optim = torch.optim.SGD(model.parameters(), lr=eta)
 
	#Suivit de l'évolutions du NN 
	#train_losses = []
	test_accuracies = []

	for n in range(nb_epochs):
		# on lit toutes les données d'apprentissage
		for x,t in train_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on met à jour les poids
			loss = loss_func(t,y)
			loss.backward()
			optim.step()
			optim.zero_grad()
			
		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for x,t in test_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on regarde si la sortie est correcte
			acc += torch.argmax(y,1) == torch.argmax(t,1)
		# on affiche le pourcentage de bonnes réponses
		test_accuracies.append((acc/data_test.shape[0]).item())
		print(acc/data_test.shape[0])
	return test_accuracies


    
if __name__ == '__main__':
	nb_epochs = 10
	for i in [32, 128] : 
		print(i)
		test_accuracies = train_return(5, 10, 0.001, i)
		# Visualisation des résultats
		plt.figure(figsize=(6, 6))

		print(test_accuracies)
		# Graphe de l'accuracy
		plt.plot(range(1, nb_epochs+1), test_accuracies, label=f'Test Accuracy : {i}', color='green')


	# Affichage des hyperparamètres choisis
	#plt.suptitle(f"Learning Rate: {eta}, Batch Size: {batch_size}, Epochs: {nb_epochs}", fontsize=14)
	plt.title('Accuracy During Testing')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	# Affichage des graphes
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()