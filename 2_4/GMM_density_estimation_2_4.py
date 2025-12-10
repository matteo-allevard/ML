from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, f1_score, precision_score, recall_score
from scipy.stats.mstats import winsorize
import re

warnings.filterwarnings('ignore')

# Charger ton dataset enrichi
print("Chargement des données...")
df = pd.read_csv("./2.4_Part2_dataset/dataset_enriched.csv")
print(f"Dataset chargé : {df.shape[0]} URLs, {df.shape[1]} colonnes")

# Séparer features et labels
features = df.drop(columns=["url", "type"])
labels = df["type"]

print(f"Nombre de features initial : {len(features.columns)}")

# =========================================
# CORRECTION 1 : PRÉ-TRAITEMENT DES VALEURS EXTRÊMES
# =========================================
print("\n" + "="*50)
print("Pré-traitement des valeurs extrêmes")
print("="*50)

# 1. Log transformation pour les comptes (évite les valeurs trop grandes)
count_features = [col for col in features.columns if col.startswith('count_')]
for col in count_features:
    features[col] = np.log1p(features[col])
    print(f"  Appliqué log1p sur {col}")

# 2. Winsorization pour limiter les extrêmes (garde les 99% centraux)
print("\nWinsorization des features numériques...")
for col in features.columns:
    if features[col].dtype in [np.float64, np.int64]:
        features[col] = winsorize(features[col], limits=[0.005, 0.005])  # 1% de chaque côté

# 3. Ajouter des features spécifiques pour détection d'anomalies
print("\nAjout de features spécifiques :")

# Détection d'IP dans l'URL (très suspect pour le phishing)
features['has_ip_in_url'] = df['url'].apply(lambda x: 1 if re.search(r'\d+\.\d+\.\d+\.\d+', str(x)) else 0)

# Densité de caractères spéciaux (phishing a souvent beaucoup de ? & =)
features['special_char_density'] = (
    features['count_equal'] + features['count_question'] + features['count_at'] + features['count_pct']
) / (features['url_length'] + 1)

# Ratio chiffres/lettres (les URLs avec beaucoup de chiffres sont suspectes)
features['digit_to_alpha_ratio'] = features['count_digit'] / (features['count_alpha'] + 1)

# Longueur relative du chemin (paths très longs = suspect)
features['path_to_url_ratio'] = features['path_length'] / (features['url_length'] + 1)

print(f"  has_ip_in_url: {features['has_ip_in_url'].sum()} URLs avec adresse IP")
print(f"  Nouveau nombre de features: {len(features.columns)}")
print(f"  Features ajoutées: has_ip_in_url, special_char_density, digit_to_alpha_ratio, path_to_url_ratio")

# =========================================
# SÉPARATION DES DONNÉES
# =========================================
print("\n" + "="*50)
print("Séparation des données")
print("="*50)

# URLs bénignes (classe normale)
benign_mask = labels == "benign"
X_benign = features[benign_mask]

# URLs malveillantes (anomalies)
anomaly_mask = labels != "benign"
X_anomaly = features[anomaly_mask]
y_anomaly = labels[anomaly_mask]

print(f"\nURLs bénignes (normales) : {len(X_benign)}")
print(f"URLs malveillantes (anomalies) : {len(X_anomaly)}")
print("Répartition des anomalies :")
print(y_anomaly.value_counts())

# =========================================
# NORMALISATION CORRECTE
# =========================================
print("\n" + "="*50)
print("Normalisation des données")
print("="*50)

# Entraîner le scaler SEULEMENT sur les données normales
scaler = StandardScaler()
Xb_scaled = scaler.fit_transform(X_benign)  # Fit sur benign seulement

# Transformer toutes les données avec le même scaler
X_all_scaled = scaler.transform(features)  # Transform, pas fit_transform!
X_anomaly_scaled = scaler.transform(X_anomaly)

# =========================================
# ENTRAÎNEMENT GMM AVEC VALIDATION
# =========================================
print("\n" + "="*50)
print("Entraînement du GMM sur données normales")
print("="*50)

# Sélection du nombre de composantes par BIC/AIC
n_components_range = range(1, 16)  # Testons jusqu'à 15
bic_scores = []
aic_scores = []

print("Calcul BIC/AIC pour différents nombres de composantes...")
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42, max_iter=200)
    gmm.fit(Xb_scaled)
    bic_scores.append(gmm.bic(Xb_scaled))
    aic_scores.append(gmm.aic(Xb_scaled))
    print(f"  Composantes {n:2d}: BIC={bic_scores[-1]:15.2f}, AIC={aic_scores[-1]:15.2f}")

# Visualisation BIC/AIC
plt.figure(figsize=(12, 6))
plt.plot(n_components_range, bic_scores, 'bo-', label='BIC', markersize=6, linewidth=2)
plt.plot(n_components_range, aic_scores, 'ro-', label='AIC', markersize=6, linewidth=2)
plt.xlabel('Nombre de composantes GMM')
plt.ylabel('Score')
plt.title('Sélection du nombre de composantes par BIC/AIC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(n_components_range)
plt.tight_layout()
plt.show()

# Choix optimal (le "coude" dans BIC)
optimal_n = np.argmin(bic_scores) + 1
print(f"\nNombre optimal de composantes (selon BIC) : {optimal_n}")

# Vérifier si on n'est pas au maximum
if optimal_n == n_components_range[-1]:
    print("ATTENTION : Le BIC n'a pas trouvé de minimum, le modèle pourrait être trop complexe!")
    optimal_n = min(10, len(X_benign) // 100)  # Limiter à 10 ou 1% des données
    print(f"  Utilisation de {optimal_n} composantes pour éviter le sur-apprentissage")

# Entraînement du modèle final
print(f"\nEntraînement du GMM final avec {optimal_n} composantes...")
gmm = GaussianMixture(n_components=optimal_n, covariance_type="full", random_state=42, max_iter=500)
gmm.fit(Xb_scaled)

# Analyser les clusters des données normales
print("\nAnalyse des clusters dans les données normales :")
cluster_labels = gmm.predict(Xb_scaled)
cluster_proportions = pd.Series(cluster_labels).value_counts().sort_index()

print("Proportion de chaque cluster :")
for cluster, count in cluster_proportions.items():
    percentage = count / len(cluster_labels) * 100
    print(f"  Cluster {cluster}: {count:7d} URLs ({percentage:5.1f}%)")

# =========================================
# SCORING DES DONNÉES
# =========================================
print("\n" + "="*50)
print("Scoring des données")
print("="*50)

# Scores de densité (log-likelihood)
scores_benign = gmm.score_samples(Xb_scaled)
scores_anomaly = gmm.score_samples(X_anomaly_scaled)

# Vérifier que les scores sont raisonnables
print(f"Vérification des scores :")
print(f"  Scores bénins - Min: {scores_benign.min():.2f}, Max: {scores_benign.max():.2f}, Mean: {scores_benign.mean():.2f}")
print(f"  Scores anomalies - Min: {scores_anomaly.min():.2f}, Max: {scores_anomaly.max():.2f}, Mean: {scores_anomaly.mean():.2f}")

# Pour visualisation : tout combiner
all_scores = np.concatenate([scores_benign, scores_anomaly])
all_labels = np.concatenate([np.zeros(len(scores_benign)), np.ones(len(scores_anomaly))])
anomaly_types = np.concatenate([['benign']*len(scores_benign), y_anomaly.values])

df["density_score"] = gmm.score_samples(X_all_scaled)

# Convertir scores en probabilités d'anomalie (plus faible score = plus anormal)
anomaly_probs = -all_scores  # Négatif car plus bas = plus anormal

# =========================================
# VISUALISATIONS
# =========================================
print("\n" + "="*50)
print("Visualisations")
print("="*50)

# Figure 1: Distribution des scores
plt.figure(figsize=(14, 10))

# Subplot 1: Histogramme général
plt.subplot(2, 2, 1)
plt.hist(scores_benign, bins=50, alpha=0.7, color='green', label='Benign', density=True)
plt.hist(scores_anomaly, bins=50, alpha=0.7, color='red', label='Anomalies', density=True)
plt.title("Distribution des scores de densité GMM")
plt.xlabel("Log-likelihood (densité)")
plt.ylabel("Densité")
plt.legend()
plt.grid(alpha=0.3)

# Subplot 2: Boxplot par type
plt.subplot(2, 2, 2)
score_df = pd.DataFrame({
    'score': all_scores,
    'type': anomaly_types,
    'is_anomaly': ['Normal' if x == 'benign' else 'Anomaly' for x in anomaly_types]
})
sns.boxplot(x='is_anomaly', y='score', data=score_df)
plt.title("Scores GMM par catégorie")
plt.xlabel("")
plt.ylabel("Log-likelihood")
plt.grid(alpha=0.3)

# Subplot 3: Densité par type d'anomalie
plt.subplot(2, 2, 3)
anomaly_only = score_df[score_df['is_anomaly'] == 'Anomaly']
sns.boxplot(x='type', y='score', data=anomaly_only, order=['defacement', 'phishing', 'malware'])
plt.title("Scores GMM par type d'anomalie")
plt.xlabel("Type d'URL malveillante")
plt.ylabel("Log-likelihood")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

# Subplot 4: Courbe ROC
plt.subplot(2, 2, 4)
fpr, tpr, thresholds = roc_curve(all_labels, anomaly_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'Courbe ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Détection d\'anomalies')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================
# ANALYSE DES SEUILS OPTIMAUX
# =========================================
print("\n" + "="*50)
print("Analyse des seuils de détection")
print("="*50)

# Trouver le seuil qui maximise F1-score
print("Recherche du seuil optimal...")
f1_scores = []
thresholds_to_test = np.percentile(-scores_benign, np.arange(5, 100, 2.5))

for threshold in thresholds_to_test:
    predictions = (anomaly_probs > threshold).astype(int)
    f1 = f1_score(all_labels, predictions)
    f1_scores.append((threshold, f1))

best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
print(f"Meilleur seuil (F1 optimal) : {best_threshold:.2f}")
print(f"F1-score optimal : {best_f1:.3f}")

# Tester différents percentiles
print("\nPerformance avec différents seuils :")
percentiles = [85, 90, 92, 95, 97, 99]
results = []

for p in percentiles:
    threshold = np.percentile(-scores_benign, p)
    predictions = (anomaly_probs > threshold).astype(int)
    
    precision = precision_score(all_labels, predictions, pos_label=1)
    recall = recall_score(all_labels, predictions, pos_label=1)
    f1 = f1_score(all_labels, predictions, pos_label=1)
    
    # Calculer les vrais/faux positifs/négatifs
    tn = np.sum((predictions == 0) & (all_labels == 0))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    tp = np.sum((predictions == 1) & (all_labels == 1))
    
    results.append({
        'percentile': p,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })
    
    print(f"\nSeuil {p}%-ile ({threshold:.2f}):")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"  VP: {tp:,d}  FP: {fp:,d}  FN: {fn:,d}  VN: {tn:,d}")
    print(f"  Anomalies détectées: {tp}/{tp+fn} ({recall*100:.1f}%)")
    print(f"  Faux positifs: {fp}/{fp+tn} ({(fp/(fp+tn)*100 if fp+tn>0 else 0):.1f}%)")

# Choisir le seuil avec le meilleur compromis
# On veut un bon recall sans trop de faux positifs
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("Résumé des performances par seuil :")
print("="*50)
print(results_df[['percentile', 'precision', 'recall', 'f1', 'tp', 'fp']].to_string())

# Sélectionner le seuil à 92% (bon compromis)
selected_percentile = 92
selected_threshold = results_df[results_df['percentile'] == selected_percentile]['threshold'].values[0]
predictions_selected = (anomaly_probs > selected_threshold).astype(int)

print(f"\n" + "="*50)
print(f"Rapport de classification (seuil {selected_percentile}e percentile):")
print("="*50)
print(classification_report(all_labels, predictions_selected, 
                           target_names=['Normal', 'Anomaly'],
                           digits=3))

# Ajouter les prédictions au dataframe
df['is_anomaly_pred'] = (anomaly_probs > selected_threshold).astype(int)

# =========================================
# ANALYSE DÉTAILLÉE DES RÉSULTATS
# =========================================
print("\n" + "="*50)
print("Analyse détaillée des résultats")
print("="*50)

# Faux négatifs (anomalies non détectées)
false_negatives = df[(df['type'] != 'benign') & (df['is_anomaly_pred'] == 0)]
false_positives = df[(df['type'] == 'benign') & (df['is_anomaly_pred'] == 1)]
true_positives = df[(df['type'] != 'benign') & (df['is_anomaly_pred'] == 1)]

print(f"\nVrais positifs : {len(true_positives):,d} anomalies détectées")
print(f"Faux négatifs : {len(false_negatives):,d} anomalies non détectées")
print(f"Faux positifs : {len(false_positives):,d} normales classées comme anomalies")

print("\nRépartition des vrais positifs par type :")
print(true_positives['type'].value_counts())

print("\nRépartition des faux négatifs par type :")
print(false_negatives['type'].value_counts())

# Taux de détection par type
print("\nTaux de détection par type d'anomalie :")
for anomaly_type in ['defacement', 'phishing', 'malware']:
    total = (df['type'] == anomaly_type).sum()
    detected = ((df['type'] == anomaly_type) & (df['is_anomaly_pred'] == 1)).sum()
    detection_rate = detected / total * 100
    print(f"  {anomaly_type:10s}: {detected:6,d}/{total:6,d} ({detection_rate:5.1f}%)")

# =========================================
# IDENTIFICATION DES POINTS LES PLUS ANORMAUX
# =========================================
print("\n" + "="*50)
print("Top 10 URLs les plus anormales")
print("="*50)

# Chercher les scores les PLUS BAS (plus anormaux)
top_anomalies = df.nsmallest(20, 'density_score')

print(f"Analyse des {len(top_anomalies)} URLs les plus anormales :")
print(f"  Répartition par type: {top_anomalies['type'].value_counts().to_dict()}")

for idx, row in top_anomalies.head(10).iterrows():
    print(f"\n{'='*60}")
    print(f"URL: {row['url'][:120]}...")
    print(f"Type réel: {row['type']}")
    print(f"Score GMM: {row['density_score']:.2f}")
    print(f"Prédiction: {'✅ ANOMALIE' if row['is_anomaly_pred'] == 1 else '❌ NORMAL'}")
    
    # Identifier les features les plus extrêmes
    url_features = features.loc[idx]
    
    # Chercher les 5 features les plus éloignées de la moyenne
    feature_means = features.mean()
    deviations = abs(url_features - feature_means)
    top_features = deviations.nlargest(5).index
    
    print(f"\nFeatures les plus anormales :")
    for feat in top_features:
        value = url_features[feat]
        mean = feature_means[feat]
        std = features[feat].std()
        z_score = (value - mean) / std if std > 0 else 0
        print(f"  {feat:25s}: {value:8.2f} (moyenne: {mean:6.2f}, z-score: {z_score:6.2f})")

# =========================================
# ANALYSE DES CLUSTERS DES DONNÉES NORMALES
# =========================================
print("\n" + "="*50)
print("Analyse des clusters des URLs normales")
print("="*50)

# Obtenir les clusters pour toutes les données normales
benign_clusters = gmm.predict(Xb_scaled)

# Créer un dataframe pour analyser les clusters
benign_df = X_benign.copy()
benign_df['cluster'] = benign_clusters

# Analyser chaque cluster
print("\nCaractéristiques des clusters normaux :")
for cluster in range(optimal_n):
    cluster_data = benign_df[benign_df['cluster'] == cluster]
    if len(cluster_data) > 100:  # Ignorer les très petits clusters
        print(f"\nCluster {cluster} ({len(cluster_data):,d} URLs, {len(cluster_data)/len(benign_df)*100:.1f}%) :")
        
        # Moyennes des principales features
        print(f"  url_length moyen: {cluster_data['url_length'].mean():.1f}")
        print(f"  has_https: {cluster_data['has_https'].mean()*100:.1f}%")
        print(f"  count_digit moyen: {cluster_data['count_digit'].mean():.1f}")
        print(f"  has_ip_in_url: {cluster_data['has_ip_in_url'].mean()*100:.1f}%")
        print(f"  special_char_density: {cluster_data['special_char_density'].mean():.3f}")

# =========================================
# ANALYSE DES FAUX NÉGATIFS (POUR AMÉLIORATION)
# =========================================
print("\n" + "="*50)
print("Analyse des faux négatifs pour amélioration")
print("="*50)

# Analyser les caractéristiques des faux négatifs par type
for anomaly_type in ['phishing', 'defacement', 'malware']:
    fn_type = false_negatives[false_negatives['type'] == anomaly_type]
    if len(fn_type) > 0:
        print(f"\n{anomaly_type.upper()} - {len(fn_type):,d} non détectés :")
        
        # Prendre un échantillon pour analyse
        sample = fn_type.sample(min(5, len(fn_type)), random_state=42)
        
        for _, row in sample.iterrows():
            print(f"\n  Exemple: {row['url'][:80]}...")
            print(f"  Score: {row['density_score']:.2f}")
            
            # Vérifier les features spécifiques
            if anomaly_type == 'phishing':
                url_features = features.loc[row.name]
                if url_features['has_ip_in_url'] > 0:
                    print(f"  → CONTIENT une IP mais pas détecté!")
                if url_features['special_char_density'] > 0.1:
                    print(f"  → Densité caractères spéciaux: {url_features['special_char_density']:.3f}")

# =========================================
# EXPORT DES RÉSULTATS
# =========================================
print("\n" + "="*50)
print("Export des résultats")
print("="*50)

# Ajouter des colonnes supplémentaires pour l'analyse
df['anomaly_score'] = -df['density_score']  # Score positif (plus élevé = plus anormal)

# Calculer le percentile de chaque score par rapport aux normales
percentiles = []
for score in df['density_score']:
    percentile = np.sum(scores_benign <= score) / len(scores_benign) * 100
    percentiles.append(percentile)
df['normal_percentile'] = percentiles

# Ajouter le cluster prédit pour les URLs normales
df['gmm_cluster'] = -1  # -1 pour les anomalies
df.loc[benign_mask, 'gmm_cluster'] = gmm.predict(Xb_scaled)

# Sauvegarder
output_path = "./2.4_Part2_dataset/dataset_with_anomaly_scores_improved.csv"
df.to_csv(output_path, index=False)
print(f"\nDataset avec scores sauvegardé dans '{output_path}'")
print(f"Colonnes ajoutées :")
print(f"  - density_score : log-likelihood du GMM")
print(f"  - anomaly_score : score d'anomalie (plus haut = plus anormal)")
print(f"  - is_anomaly_pred : prédiction binaire (seuil {selected_percentile}e percentile)")
print(f"  - normal_percentile : percentile par rapport aux URLs normales")
print(f"  - gmm_cluster : cluster GMM attribué (-1 pour anomalies)")

# =========================================
# RÉSUMÉ FINAL ET RECOMMANDATIONS
# =========================================
print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)
print(f"Données totales : {len(df):,d} URLs")
print(f"URLs normales (benign) : {len(X_benign):,d}")
print(f"URLs anomalies : {len(X_anomaly):,d}")

# Calcul final des métriques
final_precision = precision_score(all_labels, predictions_selected, pos_label=1)
final_recall = recall_score(all_labels, predictions_selected, pos_label=1)
final_f1 = f1_score(all_labels, predictions_selected, pos_label=1)

print(f"\nPerformance GMM avec {optimal_n} composantes :")
print(f"  AUC ROC : {roc_auc:.3f}")
print(f"  Precision anomalies : {final_precision:.3f}")
print(f"  Recall anomalies : {final_recall:.3f}")
print(f"  F1-score anomalies : {final_f1:.3f}")

print(f"\nDétection par type (avec seuil {selected_percentile}%) :")
for anomaly_type in ['defacement', 'phishing', 'malware']:
    total = (df['type'] == anomaly_type).sum()
    detected = ((df['type'] == anomaly_type) & (df['is_anomaly_pred'] == 1)).sum()
    rate = detected / total * 100
    print(f"  {anomaly_type:10s}: {detected:6,d}/{total:6,d} ({rate:5.1f}%)")

print("\n" + "="*50)
print("RECOMMANDATIONS POUR AMÉLIORATION")
print("="*50)
print("1. Les valeurs extrêmes sont maintenant contrôlées (winsorization + log)")
print("2. Features spécifiques ajoutées (IP dans URL, densité caractères spéciaux)")
print("3. Seuil ajusté pour meilleur compromis précision/recall")
print("4. Pour améliorer encore :")
print("   - Entraîner des modèles séparés par type de menace")
print("   - Ajouter plus de features lexicales (mots-clés de phishing)")
print("   - Combiner avec Isolation Forest ou One-Class SVM")
print("   - Utiliser les scores GMM comme features pour un classifieur supervisé")

print("\n" + "="*50)
print("PROCHAINES ÉTAPES SUGGÉRÉES")
print("="*50)
print("1. Tester Isolation Forest sur les mêmes données")
print("2. Combiner les scores GMM et Isolation Forest")
print("3. Entraîner XGBoost avec toutes les features + scores d'anomalie")
print("4. Analyser les faux positifs pour affiner le modèle")

print("\n✅ Analyse GMM terminée avec succès!")