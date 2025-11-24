# =============================================================
# PI2 - MACHINE LEARNING NÃO SUPERVISIONADO
# Tema: Agrupamento de músicas por características
# Modelos: K-Means + DBSCAN
#
# Objetivo:
# Descobrir perfis de músicas e sugerir playlists automáticas
# com base em BPM, energia, ritmos, dançabilidade e duração.
# =============================================================


# 1) IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN


# 2) ETL – CRIAÇÃO E LIMPEZA DOS DADOS

# ---------- Dados fictícios para as músicas ----------
musicas = pd.DataFrame({
    "BPM": [120, 128, 90, 100, 140, 75, 160, 130, 110, 95, 150, 80],
    "energia": [0.8, 0.9, 0.4, 0.6, 0.95, 0.3, 0.98, 0.85, 0.7, 0.5, 0.99, 0.35],
    "dancabilidade": [0.7, 0.85, 0.5, 0.6, 0.9, 0.3, 0.95, 0.8, 0.65, 0.55, 0.92, 0.4],
    "duracao_seg": [210, 180, 240, 200, 190, 260, 175, 185, 220, 230, 195, 250]
})

print("\n===== DADOS DAS MÚSICAS =====")
print(musicas.head())

# Padronização dos dados
scaler = StandardScaler()
musicas_normalizadas = scaler.fit_transform(musicas)

# --------------------------------------------------------------


# 3) VISUALIZAÇÕES DOS DADOS

# ---------- Gráfico 1: BPM x Energia ----------
plt.figure(figsize=(6,4))
plt.scatter(musicas["BPM"], musicas["energia"])
plt.xlabel("BPM")
plt.ylabel("Energia")
plt.title("BPM x Energia")
plt.grid()
plt.show()

# ---------- Gráfico 2: Dançabilidade x Energia ----------
plt.figure(figsize=(6,4))
plt.scatter(musicas["dancabilidade"], musicas["energia"])
plt.xlabel("Dançabilidade")
plt.ylabel("Energia")
plt.title("Dançabilidade x Energia")
plt.grid()
plt.show()

# ---------- Gráfico 3: Duração x BPM ----------
plt.figure(figsize=(6,4))
plt.scatter(musicas["duracao_seg"], musicas["BPM"])
plt.xlabel("Duração (segundos)")
plt.ylabel("BPM")
plt.title("Duração x BPM")
plt.grid()
plt.show()

# Ritmos musicais
musicas["ritmo"] = [
    "Samba", "Pagode", "Rock", "MPB", "Eletrônica", "Reggae",
    "Funk", "Pop", "Rock", "MPB", "Eletrônica", "Samba"
]

print("\n===== RITMOS ADICIONADOS =====")
print(musicas[["BPM", "energia", "dancabilidade", "ritmo"]])

# ---------- Gráfico: BPM x Energia com labels de ritmo ----------
plt.figure(figsize=(8,6))
plt.scatter(musicas["BPM"], musicas["energia"], s=100)

# Nome do ritmo em cada ponto
for i in range(len(musicas)):
    plt.text(
        musicas["BPM"][i] + 0.5,
        musicas["energia"][i] + 0.005,
        musicas["ritmo"][i],
        fontsize=10
    )

plt.xlabel("BPM")
plt.ylabel("Energia")
plt.title("Distribuição dos ritmos por BPM e Energia")
plt.grid(True)
plt.show()


# --------------------------------------------------------------


# 4) MODELO 1 – K-MEANS

print("\n==============================")
print(" MODELO 1 – K-MEANS")
print("==============================\n")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(musicas_normalizadas)

musicas["cluster_kmeans"] = clusters_kmeans
print("Clusters identificados pelo K-Means:")
print(musicas[["BPM", "energia", "dancabilidade", "duracao_seg", "cluster_kmeans"]])

# ---------- Gráfico K-Means ----------
plt.figure(figsize=(6,4))
plt.scatter(musicas["BPM"], musicas["energia"], c=clusters_kmeans)
plt.xlabel("BPM")
plt.ylabel("Energia")
plt.title("K-Means – Agrupamento de Músicas")
plt.grid()
plt.show()

# --------------------------------------------------------------


# 5) MODELO 2 – DBSCAN

print("\n==============================")
print(" MODELO 2 – DBSCAN")
print("==============================\n")

dbscan = DBSCAN(eps=1.2, min_samples=2)
clusters_dbscan = dbscan.fit_predict(musicas_normalizadas)

musicas["cluster_dbscan"] = clusters_dbscan
print("Clusters identificados pelo DBSCAN:")
print(musicas[["BPM", "energia", "dancabilidade", "duracao_seg", "cluster_dbscan"]])

# ---------- Gráfico DBSCAN ----------
plt.figure(figsize=(6,4))
plt.scatter(musicas["BPM"], musicas["energia"], c=clusters_dbscan)
plt.xlabel("BPM")
plt.ylabel("Energia")
plt.title("DBSCAN – Agrupamento de Músicas")
plt.grid()
plt.show()


# --------------------------------------------------------------


# 6) ANÁLISE E INTERPRETAÇÃO DOS RESULTADOS

print("\n===== INTERPRETAÇÃO DOS RESULTADOS =====\n")

# INTERPRETAÇÃO DO MODELO 1 – K-MEANS
print("MODELO 1 – K-Means (Perfil das Músicas)\n")
print("""
O K-Means dividiu as músicas em 3 grupos.
Isso significa que existem 3 perfis principais, por exemplo:

► Cluster 0: músicas mais calmas, BPM menor, energia baixa.
► Cluster 1: músicas equilibradas, BPM médio e energia moderada.
► Cluster 2: músicas muito energéticas, BPM alto e alta dançabilidade.

Esse tipo de agrupamento pode ser usado para:
- Criar playlists automáticas
- Sugerir músicas semelhantes
- Identificar estilos dominantes
""")

# INTERPRETAÇÃO DO MODELO 2 – DBSCAN
print("\nMODELO 2 – DBSCAN (Detecção de Padrões Naturais)\n")
print("""
O DBSCAN identifica grupos baseados na densidade e pode reconhecer músicas
que fogem do padrão (classificando-as como -1, que significa "ruído").

Ele pode encontrar grupos que não são esféricos como no K-Means.

Isso é útil para:
- Detectar músicas muito diferentes do resto
- Identificar estilos bem específicos
- Separar outliers automaticamente
""")

print("\nProjeto executado com sucesso!")
