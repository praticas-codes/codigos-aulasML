import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.cluster import KMeans

# 1. Gerando dados fictícios que naturalmente formam 3 grupos (mas a IA não sabe disso!)
X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.60, random_state=42)

# Ajustando a escala para parecer Renda (em milhares) e Gastos (de 0 a 100)
X[:, 0] = (X[:, 0] - X[:, 0].min()) * 10 + 20  # Renda
X[:, 1] = (X[:, 1] - X[:, 1].min()) * 15 + 10  # Gastos

# Criando um DataFrame para facilitar
df_shopping = pd.DataFrame(X, columns=['Renda Anual (R$ Mil)', 'Pontuação de Gastos (0-100)'])

# 2. Plotando o gráfico sem grupos (Tudo cinza!)
plt.figure(figsize=(9, 6))
plt.scatter(df_shopping['Renda Anual (R$ Mil)'], df_shopping['Pontuação de Gastos (0-100)'], c='darkgray', s=80, edgecolor='black')

plt.title("Dados Brutos dos Clientes (O que a IA recebe)", fontsize=14)
plt.xlabel("Renda Anual (R$ Mil de Reais)", fontsize=12)
plt.ylabel("Pontuação de Gastos (0 a 100)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# 1. Criando o algoritmo e pedindo 3 grupos (caixa-preta por enquanto!)
algoritmo_cluster = KMeans(n_clusters=3, random_state=42)
df_shopping['Grupo_IA'] = algoritmo_cluster.fit_predict(X)

# 2. Plotando novamente, mas agora colorindo os pontos com as respostas da IA
plt.figure(figsize=(9, 6))
plt.scatter(
    df_shopping['Renda Anual (R$ Mil)'], 
    df_shopping['Pontuação de Gastos (0-100)'], 
    c=df_shopping['Grupo_IA'], 
    cmap='viridis', 
    s=80, 
    edgecolor='black'
)

plt.title("Dados Organizados pela IA (Padrões Descobertos)", fontsize=14)
plt.xlabel("Renda Anual (R$ Mil de Reais)", fontsize=12)
plt.ylabel("Pontuação de Gastos (0 a 100)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


