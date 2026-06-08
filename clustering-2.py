#célula 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Gerando os dados fictícios
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.60, random_state=42)

# 2. Vamos forçar o KMeans a rodar apenas ONCE (max_iter=1) com inicialização aleatória
# para mostrar para os alunos o "Passo 2" e "Passo 3" do algoritmo
kmeans_inicial = KMeans(n_clusters=3, init='random', max_iter=1, n_init=1, random_state=1)
labels_iniciais = kmeans_inicial.fit_predict(X)
centroides_iniciais = kmeans_inicial.cluster_centers_

# Plotando o estado inicial do algoritmo
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_iniciais, cmap='viridis', s=60, alpha=0.5, edgecolor='k', label='Dados Associados')
plt.scatter(centroides_iniciais[:, 0], centroides_iniciais[:, 1], c='red', marker='*', s=300, edgecolor='black', label='Centróides Iniciais')

plt.title("K-Means: Passo Inicial (Centróides jogados ao acaso e primeira associação)", fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

#copiar célula 2
# 1. Deixando o algoritmo rodar até o final (padrão do Scikit-Learn)
kmeans_final = KMeans(n_clusters=3, random_state=42)
labels_finais = kmeans_final.fit_predict(X)
centroides_finais = kmeans_final.cluster_centers_

# Plotando o estado final estabilizado
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_finais, cmap='viridis', s=60, edgecolor='k')
plt.scatter(centroides_finais[:, 0], centroides_finais[:, 1], c='red', marker='*', s=350, edgecolor='black', label='Centróides Finais (O Centro Real)')

plt.title("K-Means: Estado de Convergência (Centróides encontraram o centro perfeito)", fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
