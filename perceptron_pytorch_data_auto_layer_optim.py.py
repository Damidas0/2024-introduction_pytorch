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
    
if __name__ == '__main__':
	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	eta = 0.00001 # taux d'apprentissage
	
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	# on initialise le modèle et ses poids
	# model = torch.nn.Linear(data_train.shape[1],label_train.shape[1])
	model = ShallowNet()
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
  
      # Visualisation des résultats
	plt.figure(figsize=(6, 6))

	print(test_accuracies)
	# Graphe de l'accuracy
	plt.plot(range(1, nb_epochs+1), test_accuracies, label="Test Accuracy", color='green')
	plt.title('Accuracy During Testing')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()

	# Affichage des hyperparamètres choisis
	plt.suptitle(f"Learning Rate: {eta}, Batch Size: {batch_size}, Epochs: {nb_epochs}", fontsize=14)

	# Affichage des graphes
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()