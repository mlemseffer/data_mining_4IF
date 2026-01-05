import re
import pandas as pd

# Stopwords français et anglais (liste manuelle)
stop_words = {
    # Anglais
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't",
    # Français
    'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 
    'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 
    'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 
    'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 
    'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'suis', 
    'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 
    'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 
    'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 
    'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'eu', 'eue', 'eues', 'eus', 
    'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 
    'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 
    'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 
    'eût', 'eussions', 'eussiez', 'eussent', 'ceci', 'cela', 'celà', 'cet', 'cette', 'ici', 'ils', 
    'les', 'leurs', 'quel', 'quels', 'quelle', 'quelles', 'sans', 'soi'
}

custom_stopwords = {
    'camera', 'canon', 'des', 'digital', 'europe', 'flickr', 'flickriosapp', 
    'flickrmobile', 'francia', 'france', 'image', 'images', 'img', 
    'instagram', 'lyon', 'nikon', 'photo', 'photos', 'picture', 
    'pictures', 'shot', 'taken', 'uploaded'
}

def simple_stem(word):
    """Stemming simple pour français et anglais"""
    # Suffixes anglais courants
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('ly'):
        return word[:-2]
    if word.endswith('ness'):
        return word[:-4]
    if word.endswith('ment'):
        return word[:-4]
    # Suffixes français courants
    if word.endswith('tion'):
        return word[:-4]
    if word.endswith('sion'):
        return word[:-4]
    if word.endswith('able'):
        return word[:-4]
    if word.endswith('ible'):
        return word[:-4]
    if word.endswith('ique'):
        return word[:-4]
    if word.endswith('eur'):
        return word[:-3]
    if word.endswith('euse'):
        return word[:-4]
    if word.endswith('ait'):
        return word[:-3]
    if word.endswith('aient'):
        return word[:-6]
    if word.endswith('er'):
        return word[:-2]
    if word.endswith('é'):
        return word[:-1]
    if word.endswith('és'):
        return word[:-2]
    if word.endswith('ées'):
        return word[:-3]
    return word

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    # Tokenisation simple par split
    tokens = text.split()
    tokens = [t.strip() for t in tokens if t.strip()]
    tokens = [t for t in tokens if t not in stop_words and t not in custom_stopwords and len(t) > 2]
    # Stemming simple
    tokens = [simple_stem(t) for t in tokens]
    return tokens

def preprocess_dataframe(df, text_cols=['tags', 'title']):
    for col in text_cols:
        df[f'{col}_tokens'] = df[col].apply(preprocess_text)
    return df


def compute_tfidf_per_cluster(df, cluster_col, text_cols=['tags', 'title'], top_n=10):
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
    
    # Calculer DF (document frequency) - dans combien de clusters apparaît chaque mot
    word_df = Counter()
    for cluster_id, tokens in cluster_texts.items():
        unique_words = set(tokens)
        for word in unique_words:
            word_df[word] += 1
    
    # Calculer TF-IDF pour chaque cluster
    cluster_tfidf = {}
    for cluster_id, tokens in cluster_texts.items():
        # TF: fréquence des mots dans ce cluster
        tf = Counter(tokens)
        total_words = len(tokens)
        
        # Calculer TF-IDF pour chaque mot
        tfidf_scores = {}
        for word, count in tf.items():
            tf_normalized = count / total_words if total_words > 0 else 0
            idf = math.log(n_clusters / word_df[word]) if word_df[word] > 0 else 0
            tfidf_scores[word] = tf_normalized * idf
        
        # Trier et garder les top_n mots
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


# Exemple d'utilisation
if __name__ == '__main__':
    # Vérifier si le fichier avec clusters existe
    import os
    file_path = '../data/flickr_data2_hierarchical_sample.csv'
    
    if os.path.exists(file_path):
        print(f"Chargement de {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print(f"Le fichier {file_path} n'existe pas.")
        print("Veuillez d'abord exécuter hierarchical_clustering.py pour générer les clusters.")
        print("\nUtilisation des données nettoyées sans clusters pour démo...")
    
    # Prétraiter les textes
    print("Prétraitement des textes (tags et title)...")
    df = preprocess_dataframe(df, text_cols=['tags', 'title'])
    
    # Analyser les mots-clés pour le clustering hiérarchique (complete)
    if 'cluster_complete' in df.columns:
        print("\n=== Analyse TF-IDF pour clustering hiérarchique (complete) ===")
        cluster_keywords = compute_tfidf_per_cluster(
            df, 
            cluster_col='cluster_complete', 
            text_cols=['tags', 'title'], 
            top_n=10
        )
        display_cluster_keywords(cluster_keywords, 
                                title="Mots-clés par cluster - Hierarchical Complete")
    else:
        print("Colonne 'cluster_complete' non trouvée dans les données.")
        print("Colonnes disponibles:", df.columns.tolist())
