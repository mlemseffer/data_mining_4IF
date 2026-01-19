import re
import pandas as pd
import folium
from data_cleaning import STOP_WORDS, CUSTOM_STOPWORDS, simple_stem

# Note: Les fonctions de preprocessing (stopwords, stemming) sont maintenant dans data_cleaning.py
# et sont utilisées lors du nettoyage initial des données.


def preprocess_text(text):
    """
    Prétraite un texte pour l'analyse TF-IDF.
    Utilise les stopwords et le stemming définis dans data_cleaning.py
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    # Tokenisation simple par split
    tokens = text.split()
    tokens = [t.strip() for t in tokens if t.strip()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in CUSTOM_STOPWORDS and len(t) > 2]
    # Stemming simple
    tokens = [simple_stem(t) for t in tokens]
    return tokens

def preprocess_dataframe(df, text_cols=['text_merged']):
    """
    Prétraite le DataFrame pour l'analyse TF-IDF.
    Par défaut utilise text_merged qui est déjà nettoyé.
    
    Args:
        df: DataFrame
        text_cols: colonnes de texte à tokeniser (défaut: ['text_merged'])
    
    Returns:
        df: DataFrame avec colonnes _tokens ajoutées
    """
    for col in text_cols:
        if col in df.columns:
            df[f'{col}_tokens'] = df[col].apply(preprocess_text)
        else:
            print(f"Warning: colonne '{col}' non trouvée")
    return df


def compute_tfidf_per_cluster(df, cluster_col, text_cols=['text_merged'], top_n=10):
    """
    Calcule le TF-IDF pour chaque cluster et retourne les mots les plus pertinents.
    
    Args:
        df: DataFrame avec colonnes de texte tokenisé et labels de clusters
        cluster_col: nom de la colonne contenant les labels de clusters
        text_cols: colonnes de texte à analyser (doivent avoir des versions '_tokens')
        top_n: nombre de mots les plus pertinents à retourner par cluster
        
    Returns:
        dict: {cluster_id: [(mot, score_tfidf), ...]}
    """
    from collections import Counter
    import math
    
    # Nombre total de clusters (documents)
    n_clusters = df[cluster_col].nunique()
    
    # Pour chaque cluster, agréger tous les tokens
    cluster_texts = {}
    for cluster_id in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster_id]
        all_tokens = []
        for col in text_cols:
            token_col = f'{col}_tokens'
            if token_col in cluster_df.columns:
                for tokens in cluster_df[token_col]:
                    if isinstance(tokens, list):
                        all_tokens.extend(tokens)
        cluster_texts[cluster_id] = all_tokens
    
    word_df = Counter()
    for cluster_id, tokens in cluster_texts.items():
        unique_words = set(tokens)
        for word in unique_words:
            word_df[word] += 1
    
    cluster_tfidf = {}
    for cluster_id, tokens in cluster_texts.items():
        tf = Counter(tokens)
        total_words = len(tokens)
        
        tfidf_scores = {}
        for word, count in tf.items():
            tf_normalized = count / total_words if total_words > 0 else 0
            idf = math.log(n_clusters / word_df[word]) if word_df[word] > 0 else 0
            tfidf_scores[word] = tf_normalized * idf
        
        top_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        cluster_tfidf[cluster_id] = top_words
    
    return cluster_tfidf


def display_cluster_keywords(cluster_tfidf, title="Mots-clés par cluster (TF-IDF)"):
    """
    Affiche les mots-clés les plus pertinents pour chaque cluster.
    
    Args:
        cluster_tfidf: dict retourné par compute_tfidf_per_cluster
        title: titre de l'affichage
    """
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    
    for cluster_id in sorted(cluster_tfidf.keys()):
        print(f"\n[Cluster {cluster_id}]")
        keywords = cluster_tfidf[cluster_id]
        if keywords:
            for i, (word, score) in enumerate(keywords, 1):
                print(f"  {i}. {word:20s} (TF-IDF: {score:.4f})")
        else:
            print("  Aucun mot trouvé")
    print('='*60)


if __name__ == '__main__':
    import os
    file_path = '../data/flickr_data2_cleaned.csv'
    
    if os.path.exists(file_path):
        print(f"Chargement de {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print(f"Le fichier {file_path} n'existe pas.")
        print("Veuillez d'abord exécuter data_cleaning.py.")
        exit(1)
    
    # Prétraiter les textes (utilise text_merged par défaut)
    print("Prétraitement des textes (text_merged)...")
    df = preprocess_dataframe(df)
    
    # Si on a des clusters, analyser
    cluster_col = None
    for col in ['cluster_complete', 'cluster_hybrid', 'cluster_spatial', 'cluster']:
        if col in df.columns:
            cluster_col = col
            break
    
    if cluster_col:
        print(f"\n=== Analyse TF-IDF pour {cluster_col} ===")
        cluster_keywords = compute_tfidf_per_cluster(
            df, 
            cluster_col=cluster_col,
            top_n=10
        )
        display_cluster_keywords(cluster_keywords, 
                                title=f"Mots-clés par cluster - {cluster_col}")
    else:
        print("Aucune colonne de cluster trouvée dans les données.")
        print("Colonnes disponibles:", df.columns.tolist())
