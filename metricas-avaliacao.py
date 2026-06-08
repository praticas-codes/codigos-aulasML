import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Criando os dados reais (0 = Saudável, 1 = Doente)
# 90 saudáveis e 10 doentes
y_real = np.array([0]*90 + [1]*10)

# 2. Criando as previsões da IA para bater com o nosso exemplo:
# Para os 90 saudáveis: acertou 85 (VN) e errou 5 (FP)
# Para os 10 doentes: errou 1 (FN) e acertou 9 (VP)
y_previsao = np.array([0]*85 + [1]*5 + [0]*1 + [1]*9)

# 3. Calculando as métricas automáticas do Scikit-Learn
acuracia = accuracy_score(y_real, y_previsao)
precisao = precision_score(y_real, y_previsao)
recall = recall_score(y_real, y_previsao)

# Exibindo os resultados formatados em porcentagem
print("=== DESEMPENHO DO MODELO ===")
print(f"Acurácia geral: {acuracia * 100:.1f}%")
print(f"Precisão (Foco nos Alarmes Falsos): {precisao * 100:.1f}%")
print(f"Recall / Sensibilidade (Foco em não omitir doentes): {recall * 100:.1f}%")

import matplotlib.pyplot as plt
import seaborn as sns

# Gerando a matriz numérica de confusão
matriz = confusion_matrix(y_real, y_previsao)

# Criando textos personalizados para dentro de cada quadrante
textos_quadrantes = [
    f"Verd. Negativo (VN)\n{matriz[0][0]}\n(Previu Saudável e era)",
    f"Falso Positivo (FP)\n{matriz[0][1]}\n(Alarme Falso!)",
    f"Falso Negativo (FN)\n{matriz[1][0]}\n(Omissão Perigosa!)",
    f"Verd. Positivo (VP)\n{matriz[1][1]}\n(Previu Doente e era)"
]

# Formatando os textos para encaixar na matriz 2x2
textos_quadrantes = np.asarray(textos_quadrantes).reshape(2, 2)

# Configurando o tamanho do gráfico
plt.figure(figsize=(8, 6))

# Criando o mapa de calor (Heatmap) com Seaborn
sns.heatmap(
    matriz, 
    annot=textos_quadrantes, 
    fmt="", 
    cmap="Blues", 
    cbar=False,
    annot_kws={"size": 12, "weight": "bold"},
    linewidths=2,
    linecolor="black"
)

# Ajustando rótulos dos eixos
plt.title("Matriz de Confusão: Avaliação do Modelo Hospitalar", fontsize=14, pad=15)
plt.xlabel("Previsão do Modelo (O que a IA disse)", fontsize=12, labelpad=10)
plt.ylabel("Realidade (O que o Paciente tinha)", fontsize=12, labelpad=10)

# Nomeando as linhas e colunas
plt.xticks([0.5, 1.5], ["Saudável (0)", "Doente (1)"])
plt.yticks([0.5, 1.5], ["Saudável (0)", "Doente (1)"])

plt.show()
