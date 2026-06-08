import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Criando dados fictícios: Horas de Estudo de 15 alunos
X_horas = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 18, 20]).reshape(-1, 1)

# 2. Resultados reais: 0 = Reprovado, 1 = Aprovado
y_resultado = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Criando um DataFrame para eles visualizarem como uma tabela
tabela_alunos = pd.DataFrame({'Horas de Estudo': X_horas.flatten(), 'Aprovado?': y_resultado})
tabela_alunos.head(10)

# Criando o modelo
modelo_logistico = LogisticRegression()

# Treinando o modelo com nossos dados
modelo_logistico.fit(X_horas, y_resultado)

print("Modelo treinado com sucesso.")

# Alunos de teste
novos_alunos = np.array([[6.5], [15]])

# Pedindo as probabilidades brutas
probabilidades = modelo_logistico.predict_proba(novos_alunos)

print("Aluno 1 (6.5h de estudo):")
print(f"-> Chance de Reprovar: {probabilidades[0][0]*100:.1f}%")
print(f"-> Chance de Aprovar: {probabilidades[0][1]*100:.1f}%\n")

print("Aluno 2 (15h de estudo):")
print(f"-> Chance de Reprovar: {probabilidades[1][0]*100:.1f}%")
print(f"-> Chance de Aprovar: {probabilidades[1][1]*100:.1f}%")

# Criando uma linha contínua de horas (de 0 a 22) para desenhar a curva suavemente
horas_para_curva = np.linspace(0, 22, 100).reshape(-1, 1)
# Pegando apenas a probabilidade de aprovação (coluna 1)
probabilidades_curva = modelo_logistico.predict_proba(horas_para_curva)[:, 1]

# Plotando os alunos reais (pontos vermelhos e azuis)
plt.figure(figsize=(10, 6))
plt.scatter(X_horas, y_resultado, c=y_resultado, cmap='bwr_r', s=100, edgecolor='black', label='Alunos Reais', zorder=3)

# Plotando a curva Sigmoide (a linha do aprendizado do modelo)
plt.plot(horas_para_curva, probabilidades_curva, color='darkorange', linewidth=3, label='Curva de Probabilidade (Sigmoide)')

# Desenhando a linha de Threshold em 50% (0.5)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (50%)')

# Customização do gráfico
plt.title('Regressão Logística: Prevendo Aprovação por Horas de Estudo', fontsize=14)
plt.xlabel('Horas de Estudo', fontsize=12)
plt.ylabel('Probabilidade de Aprovação', fontsize=12)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.xlim(0, 22)

plt.show()
