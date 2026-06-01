# =========================================================
# CLASSIFICAÇÃO BINÁRIA COM SCIKIT-LEARN
# Exemplo: prever se um aluno foi aprovado ou reprovado
# =========================================================


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------
# CRIANDO UM DATASET SIMPLES
# ----------------------------------------------------
# horas_estudo -> quantidade de horas estudadas
# aprovado -> 0 = reprovado | 1 = aprovado

dados = {
    "horas_estudo": [1, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 7],
    "aprovado":     [0, 0, 0,   0, 1,   1, 1, 1,   1, 1]
}

df = pd.DataFrame(dados)

print(df)

# ----------------------------------------------------
# VISUALIZANDO OS DADOS
# ----------------------------------------------------
# Scatter plot para enxergar as classes

plt.scatter(df["horas_estudo"], df["aprovado"])

plt.xlabel("Horas de estudo")
plt.ylabel("Aprovado")
plt.title("Classificação Binária")

plt.show()

# ----------------------------------------------------
# DEFININDO FEATURES (X) E TARGET (y)
# ----------------------------------------------------
# X -> entradas
# y -> saída esperada

X = df[["horas_estudo"]]
y = df["aprovado"]

# ----------------------------------------------------
# DIVIDINDO TREINO E TESTE
# ----------------------------------------------------
# 80% treino
# 20% teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------
# CRIANDO O MODELO
# ----------------------------------------------------
# Regressão logística é MUITO usada
# em classificação binária

modelo = LogisticRegression()

# ----------------------------------------------------
# TREINANDO O MODELO
# ----------------------------------------------------

modelo.fit(X_train, y_train)

# ----------------------------------------------------
# REALIZANDO PREVISÕES
# ----------------------------------------------------

y_pred = modelo.predict(X_test)

print("Previsões:", y_pred)

# ----------------------------------------------------
# AVALIANDO O MODELO
# ----------------------------------------------------

# Acurácia
acc = accuracy_score(y_test, y_pred)

print("Acurácia:", acc)

# Matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Relatório completo
print("\nRelatório:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# TESTANDO NOVOS DADOS
# ----------------------------------------------------
# Quantas horas um aluno precisa estudar
# para provavelmente ser aprovado?

novo_aluno = [[4.5]]

previsao = modelo.predict(novo_aluno)

print("\nNovo aluno:")
print("Horas estudadas:", novo_aluno[0][0])

if previsao[0] == 1:
    print("Previsão: APROVADO")
else:
    print("Previsão: REPROVADO")

# ----------------------------------------------------
# PROBABILIDADE DA PREVISÃO
# ----------------------------------------------------
# O modelo também consegue dizer
# a probabilidade de cada classe

probabilidades = modelo.predict_proba(novo_aluno)

print("\nProbabilidades:")
print(probabilidades)

print(f"Probabilidade de reprovar: {probabilidades[0][0]:.2f}")
print(f"Probabilidade de aprovar: {probabilidades[0][1]:.2f}")
