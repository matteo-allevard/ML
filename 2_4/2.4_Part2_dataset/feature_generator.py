import pandas as pd
import numpy as np
import re
import tldextract
from urllib.parse import urlparse

# ---------------------------------------------------------
# Fonction: extraction complète de features d'une URL
# ---------------------------------------------------------
def extract_url_features(url):
    fe = {}

    # Nettoyage
    if not isinstance(url, str):
        url = ""

    parsed = urlparse(url)
    ext = tldextract.extract(url)

    # ---------------------------------------------------------
    # 1. Longueurs
    # ---------------------------------------------------------
    fe["url_length"] = len(url)
    fe["domain_length"] = len(ext.domain)
    fe["suffix_length"] = len(ext.suffix)
    fe["subdomain_length"] = len(ext.subdomain)
    fe["path_length"] = len(parsed.path)

    # ---------------------------------------------------------
    # 2. Comptages
    # ---------------------------------------------------------
    fe["count_dot"] = url.count(".")
    fe["count_slash"] = url.count("/")
    fe["count_dash"] = url.count("-")
    fe["count_equal"] = url.count("=")
    fe["count_question"] = url.count("?")
    fe["count_at"] = url.count("@")
    fe["count_pct"] = url.count("%")
    fe["count_digit"] = sum(c.isdigit() for c in url)
    fe["count_alpha"] = sum(c.isalpha() for c in url)

    # ---------------------------------------------------------
    # 3. Entropie (complexité de chaîne)
    # ---------------------------------------------------------
    def shannon_entropy(s):
        if len(s) == 0:
            return 0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in probs)

    fe["entropy"] = shannon_entropy(url)

    # ---------------------------------------------------------
    # 4. Patterns suspects
    # ---------------------------------------------------------
    fe["has_ip"] = 1 if re.match(r"^\d+\.\d+\.\d+\.\d+", ext.domain) else 0
    fe["has_https"] = 1 if parsed.scheme == "https" else 0
    fe["has_http"] = 1 if parsed.scheme == "http" else 0
    fe["has_multiple_subdomains"] = 1 if ext.subdomain.count(".") >= 1 else 0
    fe["has_login"] = 1 if "login" in url.lower() else 0
    fe["has_secure"] = 1 if "secure" in url.lower() else 0
    fe["has_update"] = 1 if "update" in url.lower() else 0
    fe["has_verify"] = 1 if "verify" in url.lower() else 0

    # ---------------------------------------------------------
    # 5. Ratio & statistiques
    # ---------------------------------------------------------
    fe["digit_ratio"] = fe["count_digit"] / fe["url_length"] if fe["url_length"] > 0 else 0
    fe["alpha_ratio"] = fe["count_alpha"] / fe["url_length"] if fe["url_length"] > 0 else 0
    fe["special_ratio"] = (fe["count_dash"] + fe["count_at"] + fe["count_pct"]) / fe["url_length"] if fe["url_length"] > 0 else 0

    return fe

# ---------------------------------------------------------
# Chargement du dataset original (2 colonnes)
# ---------------------------------------------------------
df = pd.read_csv("malicious_phish.csv")     # CHANGE LE NOM ICI SI NÉCESSAIRE
# Doit contenir: url, label

# ---------------------------------------------------------
# Extraction des features
# ---------------------------------------------------------
features = df["url"].apply(extract_url_features).apply(pd.Series)

# Fusion dataset original + features
df_final = pd.concat([df, features], axis=1)

# ---------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------
df_final.to_csv("dataset_enriched.csv", index=False)

print("Extraction terminée ! Nouveau dataset sauvegardé sous : dataset_enriched.csv")
print("Nombre de colonnes :", df_final.shape[1])
print("Aperçu :")
print(df_final.head())
