import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz vazia apenas para estruturar o grid 2x2 do slide
matriz_conceitual = np.zeros((2, 2))

# Textos explicativos mapeados exatamente nos quadrantes estatísticos corretos
textos_quadrantes = np.array([
    [
        "Verdadeiro Negativo (VN)\n\n[ ALARME SILENCIOSO ]\nNão há incêndio e o alarme ficou quieto.\n\n(Sucesso!)",
        "Falso Positivo (FP)\n\n[ ALARME FALSO ]\nNão há incêndio, mas o alarme disparou.\n\n(Inconveniente, mas seguro)"
    ],
    [
        "Falso Negativo (FN)\n\n[ A OMISSÃO ]\nA casa está pegando fogo e o alarme NÃO tocou!\n\n(ERRO CATASTRÓFICO!)",
        "Verdadeiro Positivo (VP)\n\n[ ALARME TOCOU ]\nHá incêndio e o alarme disparou.\n\n(Sucesso!)"
    ]
])

# Configura o tamanho ideal para apresentações widescreen (slides)
plt.figure(figsize=(11, 7))

# Criando o placar visual com Seaborn
sns.heatmap(
    matriz_conceitual,
    annot=textos_quadrantes,
    fmt="",
    cmap="Pastel1",  # Cores suaves e profissionais para slides
    cbar=False,      # Remove a barra de intensidade numérica
    annot_kws={"size": 12, "weight": "bold", "color": "black"},
    linewidths=5,
    linecolor="white"
)

# Títulos e formatação dos eixos
plt.title("O Placar da Tomada de Decisão: Matriz de Confusão", fontsize=16, pad=25, weight="bold")
plt.xlabel("Comportamento do Alarme (Previsão do Modelo)", fontsize=13, labelpad=15, weight="bold")
plt.ylabel("Realidade da Casa (Classe Real)", fontsize=13, labelpad=15, weight="bold")

# Substituindo os números 0 e 1 por rótulos intuitivos
plt.xticks([0.5, 1.5], ["Ficou Silencioso (0)", "Disparou / Tocou (1)"], fontsize=12)
plt.yticks([0.5, 1.5], ["Casa Sem Fogo (0)", "Casa Com Incêndio (1)"], fontsize=12, rotation=0)

plt.tight_layout()
plt.show()
