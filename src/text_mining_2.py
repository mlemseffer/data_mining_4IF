import re
import pandas as pd
import numpy as np
from collections import Counter
from data_cleaning import STOP_WORDS, CUSTOM_STOPWORDS, simple_stem

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠ rank_bm25 non installé. Installez avec: pip install rank-bm25")


def preprocess_text(text):
    """
    Prétraite un texte pour l'analyse BM25.
    Utilise les stopwords et le stemming définis dans data_cleaning.py
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    tokens = text.split()
    tokens = [t.strip() for t in tokens if t.strip()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in CUSTOM_STOPWORDS and len(t) > 2]
    tokens = [simple_stem(t) for t in tokens]
    return tokens


def preprocess_dataframe(df, text_cols=['text_merged']):
    """
    Prétraite le DataFrame pour l'analyse BM25.
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


def compute_bm25_per_cluster(df, cluster_col, text_cols=['text_merged'], top_n=10):
    """
    Calcule le BM25 pour chaque cluster et retourne les mots les plus pertinents.
    BM25 est plus performant que TF-IDF car il prend en compte la longueur des documents.
    
    Args:
        df: DataFrame avec colonnes de texte tokenisé et labels de clusters
        cluster_col: nom de la colonne contenant les labels de clusters
        text_cols: colonnes de texte à analyser (doivent avoir des versions '_tokens')
        top_n: nombre de mots les plus pertinents à retourner par cluster
        
    Returns:
        dict: {cluster_id: [(mot, score_bm25), ...]}
    """
    if not BM25_AVAILABLE:
        print("❌ BM25 non disponible, impossible de calculer les scores")
        return {}
    
    cluster_keywords = {}
    
    for cluster_id in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster_id]
        
        corpus = []
        for col in text_cols:
            token_col = f'{col}_tokens'
            if token_col in cluster_df.columns:
                for tokens in cluster_df[token_col]:
                    if isinstance(tokens, list) and len(tokens) > 0:
                        corpus.append(tokens)
        
        if len(corpus) == 0:
            cluster_keywords[cluster_id] = []
            continue
        
        bm25 = BM25Okapi(corpus)
        
        all_words = set([word for doc in corpus for word in doc])
        word_scores = {}
        
        for word in all_words:
            scores = bm25.get_scores([word])
            word_scores[word] = scores.mean()
        
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        cluster_keywords[cluster_id] = top_words
    
    return cluster_keywords


def compute_bm25_features(df, text_cols=['text_merged'], max_features=100):
    """
    Calcule les features BM25 pour tout le dataset.
    Retourne une matrice de features pour le clustering hybride.
    
    Args:
        df: DataFrame avec colonnes de texte tokenisé
        text_cols: colonnes de texte à analyser
        max_features: nombre maximum de features BM25 à retourner
    
    Returns:
        np.array: Matrice de features BM25 (n_samples, max_features)
        list: Liste des mots correspondant aux features
    """
    if not BM25_AVAILABLE:
        print("❌ BM25 non disponible")
        return np.zeros((len(df), max_features)), []
    
    corpus = []
    for idx, row in df.iterrows():
        doc_tokens = []
        for col in text_cols:
            token_col = f'{col}_tokens'
            if token_col in df.columns:
                tokens = row[token_col]
                if isinstance(tokens, list):
                    doc_tokens.extend(tokens)
        corpus.append(doc_tokens if doc_tokens else [''])
    
    bm25 = BM25Okapi(corpus)
    
    all_words_counter = Counter([word for doc in corpus for word in doc if word])
    top_words = [word for word, _ in all_words_counter.most_common(max_features)]
    
    features_matrix = np.zeros((len(corpus), len(top_words)))
    
    for i, word in enumerate(top_words):
        scores = bm25.get_scores([word])
        features_matrix[:, i] = scores
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_matrix = scaler.fit_transform(features_matrix)
    
    return features_matrix, top_words


def display_cluster_keywords(cluster_keywords, title="Mots-clés par cluster (BM25)"):
    """
    Affiche les mots-clés les plus pertinents pour chaque cluster.
    
    Args:
        cluster_keywords: dict retourné par compute_bm25_per_cluster
        title: titre de l'affichage
    """
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    
    for cluster_id in sorted(cluster_keywords.keys()):
        print(f"\n[Cluster {cluster_id}]")
        keywords = cluster_keywords[cluster_id]
        if keywords:
            for i, (word, score) in enumerate(keywords, 1):
                print(f"  {i}. {word:20s} (BM25: {score:.4f})")
        else:
            print("  Aucun mot trouvé")
    print('='*60)


def compare_with_tfidf(df, cluster_col, text_cols=['text_merged'], top_n=10):
    """
    Compare les résultats BM25 avec TF-IDF manuel.
    
    Args:
        df: DataFrame
        cluster_col: colonne de clusters
        text_cols: colonnes de texte
        top_n: nombre de mots à afficher
    """
    from text_mining import compute_tfidf_per_cluster
    
    print("\n" + "="*70)
    print("COMPARAISON BM25 vs TF-IDF")
    print("="*70)
    
    bm25_keywords = compute_bm25_per_cluster(df, cluster_col, text_cols, top_n)
    print("\n[BM25]")
    display_cluster_keywords(bm25_keywords, title="Mots-clés BM25")
    
    tfidf_keywords = compute_tfidf_per_cluster(df, cluster_col, text_cols, top_n)
    print("\n[TF-IDF]")
    from text_mining import display_cluster_keywords as display_tfidf
    display_tfidf(tfidf_keywords, title="Mots-clés TF-IDF")
    
    for cluster_id in sorted(bm25_keywords.keys()):
        bm25_words = set([w for w, _ in bm25_keywords[cluster_id]])
        tfidf_words = set([w for w, _ in tfidf_keywords.get(cluster_id, [])])
        overlap = bm25_words & tfidf_words
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Mots communs: {len(overlap)}/{top_n}")
        if overlap:
            print(f"  {', '.join(list(overlap)[:5])}")


if __name__ == '__main__':
    import os
    
    if not BM25_AVAILABLE:
        print("\n❌ Veuillez installer rank-bm25:")
        print("   pip install rank-bm25")
        exit(1)
    
    file_path = '../data/flickr_data2_cleaned.csv'
    
    if os.path.exists(file_path):
        print(f"Chargement de {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print(f"Le fichier {file_path} n'existe pas.")
        print("Veuillez d'abord exécuter data_cleaning.py.")
        exit(1)
    
    print("Prétraitement des textes (text_merged)...")
    df = preprocess_dataframe(df)
    
    cluster_col = None
    for col in ['cluster_complete', 'cluster_hybrid', 'cluster_spatial', 'cluster']:
        if col in df.columns:
            cluster_col = col
            break
    
    if cluster_col:
        print(f"\n=== Analyse BM25 pour {cluster_col} ===")
        cluster_keywords = compute_bm25_per_cluster(
            df, 
            cluster_col=cluster_col,
            top_n=10
        )
        display_cluster_keywords(cluster_keywords, 
                                title=f"Mots-clés par cluster - {cluster_col} (BM25)")
        
        choice = input("\nComparer avec TF-IDF ? (o/n) : ").strip().lower()
        if choice == 'o':
            compare_with_tfidf(df, cluster_col, top_n=10)
    else:
        print("Aucune colonne de cluster trouvée dans les données.")
        print("Colonnes disponibles:", df.columns.tolist())
