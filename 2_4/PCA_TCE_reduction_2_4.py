import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Charger ton dataset enrichi
print("Chargement des données...")
df = pd.read_csv("./2.4_Part2_dataset/dataset_enriched.csv")
print(f"Dataset chargé : {df.shape[0]} URLs, {df.shape[1]} colonnes")

# Séparer features et labels si disponibles
features = df.drop(columns=["url", "type"])
labels = df["type"] if "type" in df.columns else None

print(f"Nombre de features : {len(features.columns)}")
print(f"Features utilisées : {list(features.columns)}")

# Normalisation des features
print("\nNormalisation des features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# =========================================
# 1. PCA 2D (rapide)
# =========================================
print("\n" + "="*50)
print("1. Analyse PCA 2D")
print("="*50)

start = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - start

print(f"PCA terminé en {pca_time:.2f} secondes")
print(f"Variance expliquée : PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# Visualisation PCA
plt.figure(figsize=(10, 8))
if labels is not None:
    # Colorier par type si disponible
    unique_labels = labels.unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   s=3, alpha=0.5, label=label, color=color)
    plt.legend()
else:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=2, alpha=0.5)

plt.title(f"PCA (2D) sur {df.shape[0]} URLs\nVariance expliquée: {sum(pca.explained_variance_ratio_)*100:.1f}%")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================
# 2. OPTION A: t-SNE sur échantillon (RECOMMANDÉ)
# =========================================
print("\n" + "="*50)
print("2. Analyse t-SNE (sur échantillon)")
print("="*50)

# Échantillonnage stratifié si labels disponibles
sample_size = min(20000, len(X_scaled))  # S'assurer de ne pas dépasser la taille du dataset

if labels is not None:
    # Échantillonnage stratifié pour conserver les proportions
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(
        X_scaled, labels, 
        train_size=sample_size, 
        stratify=labels,
        random_state=42
    )
    print(f"Échantillon stratifié de {sample_size} URLs")
else:
    # Échantillonnage aléatoire simple
    indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled[indices]
    y_sample = labels[indices] if labels is not None else None
    print(f"Échantillon aléatoire de {sample_size} URLs")

# t-SNE sur l'échantillon
print("Lancement de t-SNE (cela peut prendre quelques minutes)...")
start = time.time()

tsne = TSNE(
    n_components=2,
    perplexity=min(30, sample_size // 3),  # Ajuste perplexity à la taille échantillon
    max_iter=1000,
    learning_rate='auto',
    init='pca',
    random_state=42,
    verbose=1  # Affiche la progression
)

X_tsne_sample = tsne.fit_transform(X_sample)
tsne_time = (time.time() - start) / 60

print(f"t-SNE terminé en {tsne_time:.1f} minutes")

# Visualisation t-SNE
plt.figure(figsize=(10, 8))
if y_sample is not None:
    unique_labels = np.unique(y_sample)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = y_sample == label
        plt.scatter(X_tsne_sample[mask, 0], X_tsne_sample[mask, 1], 
                   s=8, alpha=0.6, label=label, color=color)
    plt.legend()
else:
    plt.scatter(X_tsne_sample[:, 0], X_tsne_sample[:, 1], s=5, alpha=0.6)

plt.title(f"t-SNE (2D) sur échantillon de {sample_size} URLs\nTemps: {tsne_time:.1f} min")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================
# 3. OPTION B: PCA + t-SNE (alternative) - CORRIGÉ
# =========================================
print("\n" + "="*50)
print("3. Analyse PCA + t-SNE")
print("="*50)

# Réduction PCA d'abord pour accélérer
start = time.time()

# CORRECTION : Réduire à 10-15 dimensions (pas 50 car vous n'avez que 28 features)
# Garder au moins 80% de la variance
n_components_pca = min(15, X_scaled.shape[1] - 1)  # Max 15 ou moins que le nombre de features
print(f"Réduction PCA à {n_components_pca} dimensions...")

pca_reduced = PCA(n_components=n_components_pca)
X_pca_reduced = pca_reduced.fit_transform(X_scaled)

# Calculer la variance cumulée
variance_cumulee = np.cumsum(pca_reduced.explained_variance_ratio_)
print(f"Variance expliquée par {n_components_pca} composantes: {variance_cumulee[-1]*100:.1f}%")

# Prendre échantillon après PCA
if labels is not None:
    X_pca_sample, _, y_pca_sample, _ = train_test_split(
        X_pca_reduced, labels, 
        train_size=sample_size, 
        stratify=labels,
        random_state=42
    )
else:
    X_pca_sample = X_pca_reduced[indices]

# t-SNE sur données PCA réduites
tsne_pca = TSNE(
    n_components=2,
    perplexity=min(30, sample_size // 3),
    max_iter=1000,
    learning_rate='auto',
    init='random',
    random_state=42,
    verbose=1
)

X_tsne_pca = tsne_pca.fit_transform(X_pca_sample)
pca_tsne_time = (time.time() - start) / 60

print(f"PCA + t-SNE terminé en {pca_tsne_time:.1f} minutes")

# Visualisation
plt.figure(figsize=(14, 5))

# Subplot 1: PCA réduit (2 premières composantes)
plt.subplot(1, 2, 1)
if y_sample is not None:
    for label, color in zip(unique_labels, colors):
        mask = y_sample == label
        plt.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1], 
                   s=8, alpha=0.6, label=label, color=color)
else:
    plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], s=5, alpha=0.6)
plt.title(f"PCA réduit ({n_components_pca}D -> 2D)\nVariance: {variance_cumulee[1]*100:.1f}%")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.3)

# Subplot 2: PCA + t-SNE
plt.subplot(1, 2, 2)
if y_sample is not None:
    for label, color in zip(unique_labels, colors):
        mask = y_sample == label
        plt.scatter(X_tsne_pca[mask, 0], X_tsne_pca[mask, 1], 
                   s=8, alpha=0.6, label=label, color=color)
else:
    plt.scatter(X_tsne_pca[:, 0], X_tsne_pca[:, 1], s=5, alpha=0.6)
plt.title(f"PCA({n_components_pca}D) + t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(alpha=0.3)

if y_sample is not None:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle(f"Comparaison PCA réduit vs PCA+t-SNE ({sample_size} URLs)")
plt.tight_layout()
plt.show()

# =========================================
# 4. BONUS: Analyse de variance PCA complète
# =========================================
print("\n" + "="*50)
print("4. Analyse de variance PCA")
print("="*50)

# PCA complète pour voir la variance expliquée
pca_full = PCA().fit(X_scaled)
variance_cumulee_full = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(variance_cumulee_full) + 1), variance_cumulee_full * 100, 
         'bo-', linewidth=2, markersize=8)
plt.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95% variance')
plt.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% variance')

# Trouver le nombre de composantes pour 80% et 95% de variance
n_components_80 = np.argmax(variance_cumulee_full >= 0.80) + 1
n_components_95 = np.argmax(variance_cumulee_full >= 0.95) + 1

plt.axvline(x=n_components_80, color='g', linestyle=':', alpha=0.5)
plt.axvline(x=n_components_95, color='r', linestyle=':', alpha=0.5)

plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée (%)')
plt.title('Analyse de variance expliquée par PCA')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Nombre de composantes pour 80% de variance: {n_components_80}")
print(f"Nombre de composantes pour 95% de variance: {n_components_95}")
print(f"Variance totale expliquée par toutes les composantes: {variance_cumulee_full[-1]*100:.1f}%")

# =========================================
# Résumé des résultats
# =========================================
print("\n" + "="*50)
print("RÉSUMÉ")
print("="*50)
print(f"Dataset original: {df.shape[0]} URLs")
print(f"Features utilisées: {len(features.columns)}")
print(f"Temps PCA: {pca_time:.2f} secondes")
print(f"Temps t-SNE (échantillon {sample_size}): {tsne_time:.1f} minutes")
print(f"Temps PCA+t-SNE: {pca_tsne_time:.1f} minutes")
print(f"Variance totale expliquée par PCA 2D: {sum(pca.explained_variance_ratio_)*100:.1f}%")
print("\nRecommandations:")
print(f"1. Utilisez {n_components_80} composantes PCA pour garder 80% de l'information")
print("2. t-SNE est utile pour voir les clusters non-linéaires")
print("3. PCA suffit pour une visualisation rapide")