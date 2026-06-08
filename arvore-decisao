import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Criando o nosso pequeno banco de dados de alunos
dados = {
    "Faltas": [12, 2, 4, 1],
    "Entregou_Trabalho": [0, 1, 0, 1],  # 0 = Não, 1 = Sim
    "Resultado": ["Reprovado", "Aprovado", "Reprovado", "Aprovado"],
}

df_alunos = pd.DataFrame(dados)

# Separando variáveis de entrada (X) e a resposta que queremos prever (y)
X = df_alunos[["Faltas", "Entregou_Trabalho"]]
y = df_alunos["Resultado"]

# 2. Criando e treinando a Árvore de Decisão
# Usamos o critério 'gini' para ele medir a "bagunça" que explicamos acima
modelo_arvore = DecisionTreeClassifier(criterion="gini", random_state=42)
modelo_arvore.fit(X, y)

# 3. O Gráfico Clássico: Desenhando a estrutura da Árvore
plt.figure(figsize=(10, 8))

plot_tree(
    modelo_arvore,
    feature_names=["Faltas no Semestre", "Entregou Trabalho? (0=Não, 1=Sim)"],
    class_names=["Aprovado", "Reprovado"],
    filled=True,  # Pinta as caixas para podermos ver a "pureza" delas
    rounded=True,  # Deixa as caixinhas com cantos arredondados (mais bonito)
    fontsize=12,
)

plt.title("Árvore de Decisão Gerada pelo Scikit-Learn", fontsize=14, pad=20)
plt.show()
