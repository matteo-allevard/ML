import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Chargement du dataset enrichi
# ---------------------------------------------------------
df = pd.read_csv("./2.4_Part2_dataset/dataset_enriched.csv")

# ---------------------------------------------------------
# 2. Définition des features et du label
# ---------------------------------------------------------
TARGET_COLUMN = "type"
URL_COLUMN = "url"

FEATURE_COLUMNS = [
    # Longueurs
    "url_length",
    "domain_length",
    "subdomain_length",
    "suffix_length",
    "path_length",
    "query_length",

    # Comptages
    "count_dot",
    "count_slash",
    "count_dash",
    "count_at",
    "count_pct",
    "count_equal",
    "count_question",
    "count_underscore",
    "count_digit",
    "count_alpha",

    # Ratios
    "digit_ratio",
    "alpha_ratio",
    "special_ratio",

    # Entropies
    "entropy_url",
    "entropy_domain",
    "entropy_path",
    "entropy_bigram",
    "entropy_trigram",
    "transition_score",

    # Linguistique / structure
    "consonant_vowel_ratio",
    "longest_alpha_sequence",
    "path_depth",
    "num_tokens",
    "num_path_tokens",

    # Patterns
    "has_ip",
    "has_https",
    "has_http",
    "has_multiple_subdomains",

    # Mots suspects
    "has_login",
    "has_verify",
    "has_update",
    "has_secure",

    # Marques
    "brand_in_url",
    "brand_in_domain",
    "brand_in_path",

    # Paramètres
    "num_params",
    "has_long_param",

    # TLD
    "suspicious_tld",
]

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ---------------------------------------------------------
# 3. Nettoyage minimal
# ---------------------------------------------------------
X = X.fillna(0)

# ---------------------------------------------------------
# 4. Split train / test
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size :", X_train.shape)
print("Test size  :", X_test.shape)

# ---------------------------------------------------------
# 5. Entraînement du Random Forest
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,                 # un peu plus profond vu + de features
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n Entraînement du modèle...")
model.fit(X_train, y_train)
print("Entraînement terminé")

# ---------------------------------------------------------
# 6. Évaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test)

print("\n📊 Accuracy :", accuracy_score(y_test, y_pred))
print("\n📊 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 7. Importance des features
# ---------------------------------------------------------
importances = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n Feature importances :")
print(importances)

# ---------------------------------------------------------
# 8. Sauvegarde du modèle et des features
# ---------------------------------------------------------
joblib.dump(model, "random_forest_url_legacy.pkl")
joblib.dump(FEATURE_COLUMNS, "rf_features_legacy.pkl")

print("\n Modèle sauvegardé : random_forest_url_legacy.pkl")
print(" Features sauvegardées : rf_features_legacy.pkl")
