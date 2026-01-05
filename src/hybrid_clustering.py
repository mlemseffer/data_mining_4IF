import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import folium

# Import des fonctions de preprocessing depuis text_mining
from text_mining import preprocess_dataframe, compute_tfidf_per_cluster, display_cluster_keywords


def hybrid_clustering(df, n_clusters=10, spatial_weight=0.7, text_weight=0.3, random_state=42):
    """
    Clustering hybride combinant position géographique et contenu textuel.
    
    Args:
        df: DataFrame avec lat, long, tags, title
        n_clusters: nombre de clusters souhaités
        spatial_weight: poids des features spatiales (0-1)
        text_weight: poids des features textuelles (0-1)
        random_state: graine aléatoire pour reproductibilité
        
    Returns:
        df: DataFrame avec colonne 'cluster_hybrid' ajoutée
        kmeans: modèle KMeans entraîné
        vectorizer: TfidfVectorizer utilisé
        feature_info: dict avec informations sur les features
    """
    print(f"\n{'='*70}")
    print(f"CLUSTERING HYBRIDE (Spatial + Textuel)")
    print(f"{'='*70}")
    print(f"Paramètres:")
    print(f"  - Nombre de clusters: {n_clusters}")
    print(f"  - Poids spatial: {spatial_weight:.1%}")
    print(f"  - Poids textuel: {text_weight:.1%}")
    print(f"  - Nombre de photos: {len(df)}")
    
    # 1. Prétraiter les textes si pas déjà fait
    if 'tags_tokens' not in df.columns or 'title_tokens' not in df.columns:
        print("\nPrétraitement des textes...")
        df = preprocess_dataframe(df, text_cols=['tags', 'title'])
    
    # 2. Features spatiales normalisées
    print("\nExtraction des features spatiales...")
    spatial_features = df[['lat', 'long']].values
    scaler = StandardScaler()
    spatial_scaled = scaler.fit_transform(spatial_features)
    print(f"  - Features spatiales: {spatial_scaled.shape}")
    
    # 3. Features textuelles (TF-IDF)
    print("\nExtraction des features textuelles (TF-IDF)...")
    # Combiner tags et title en un seul texte
    df['combined_text'] = df.apply(
        lambda row: ' '.join(
            (row['tags_tokens'] if isinstance(row['tags_tokens'], list) else []) +
            (row['title_tokens'] if isinstance(row['title_tokens'], list) else [])
        ),
        axis=1
    )
    
    # Filtrer les textes vides
    valid_texts = df['combined_text'].apply(lambda x: len(x.strip()) > 0)
    print(f"  - Textes valides: {valid_texts.sum()} / {len(df)}")
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limiter à 100 mots les plus importants
        min_df=5,          # Mot doit apparaître dans au moins 5 documents
        max_df=0.7         # Mot ne doit pas apparaître dans plus de 70% des documents
    )
    
    text_features = vectorizer.fit_transform(df['combined_text']).toarray()
    print(f"  - Features textuelles: {text_features.shape}")
    print(f"  - Vocabulaire: {len(vectorizer.get_feature_names_out())} mots")
    
    # 4. Combiner les features avec pondération
    print("\nCombinaison des features...")
    combined_features = np.hstack([
        spatial_scaled * spatial_weight,
        text_features * text_weight
    ])
    print(f"  - Features combinées: {combined_features.shape}")
    
    # 5. Clustering K-means
    print(f"\nClustering K-means (k={n_clusters})...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=300
    )
    df['cluster_hybrid'] = kmeans.fit_predict(combined_features)
    
    # 6. Évaluation
    silhouette_avg = silhouette_score(combined_features, df['cluster_hybrid'])
    print(f"\nScore de silhouette: {silhouette_avg:.3f}")
    
    # 7. Statistiques par cluster
    print(f"\n{'='*70}")
    print("STATISTIQUES PAR CLUSTER")
    print(f"{'='*70}")
    print(f"{'Cluster':<10} {'Taille':<10} {'Lat moy':<12} {'Long moy':<12} {'% du total':<12}")
    print('-'*70)
    
    for cluster_id in sorted(df['cluster_hybrid'].unique()):
        cluster_df = df[df['cluster_hybrid'] == cluster_id]
        size = len(cluster_df)
        lat_mean = cluster_df['lat'].mean()
        long_mean = cluster_df['long'].mean()
        percentage = (size / len(df)) * 100
        print(f"{cluster_id:<10} {size:<10} {lat_mean:<12.4f} {long_mean:<12.4f} {percentage:<12.1f}%")
    
    print('-'*70)
    print(f"Taille min: {df['cluster_hybrid'].value_counts().min()}")
    print(f"Taille max: {df['cluster_hybrid'].value_counts().max()}")
    print(f"Taille moyenne: {df['cluster_hybrid'].value_counts().mean():.1f}")
    print(f"Écart-type: {df['cluster_hybrid'].value_counts().std():.1f}")
    
    # Informations sur les features
    feature_info = {
        'scaler': scaler,
        'vectorizer': vectorizer,
        'spatial_weight': spatial_weight,
        'text_weight': text_weight,
        'vocabulary': vectorizer.get_feature_names_out().tolist(),
        'n_spatial_features': spatial_scaled.shape[1],
        'n_text_features': text_features.shape[1]
    }
    
    return df, kmeans, vectorizer, feature_info


def visualize_hybrid_clusters(df, output_file='../maps/clusters_hybrid.html', sample_size=2000):
    """
    Visualise les clusters hybrides sur une carte interactive de Lyon.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_hybrid'
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher (pour la performance)
    """
    print(f"\n{'='*70}")
    print("VISUALISATION DES CLUSTERS SUR CARTE")
    print(f"{'='*70}")
    
    # Calculer les mots-clés par cluster
    print("Calcul des mots-clés TF-IDF par cluster...")
    cluster_keywords = compute_tfidf_per_cluster(
        df,
        cluster_col='cluster_hybrid',
        text_cols=['tags', 'title'],
        top_n=3
    )
    
    # Créer la carte centrée sur Lyon
    center_lat = df['lat'].mean()
    center_long = df['long'].mean()
    m = folium.Map(location=[center_lat, center_long], zoom_start=12)
    
    # Couleurs pour les clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'gray', 'black', 'lightgray', 'white', 'brown']
    
    # Échantillonner si trop de points
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    print(f"Affichage de {len(df_sample):,} points sur {len(df):,}")
    
    # Ajouter les points colorés par cluster
    for idx, row in df_sample.iterrows():
        cluster_id = int(row['cluster_hybrid'])
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=colors[cluster_id % len(colors)],
            fill=True,
            fillColor=colors[cluster_id % len(colors)],
            fillOpacity=0.6,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)
    
    # Ajouter les centres des clusters
    for cluster_id in sorted(df['cluster_hybrid'].unique()):
        cluster_df = df[df['cluster_hybrid'] == cluster_id]
        center_lat = cluster_df['lat'].mean()
        center_long = cluster_df['long'].mean()
        
        # Récupérer les 3 mots-clés les plus pertinents
        keywords = cluster_keywords.get(cluster_id, [])
        keywords_text = '<br>'.join([f"{i+1}. {word}" for i, (word, score) in enumerate(keywords)])
        
        popup_html = f"""<b>Cluster {cluster_id}</b><br>
        Taille: {len(cluster_df)} photos<br>
        <br><b>Mots-clés:</b><br>
        {keywords_text if keywords_text else 'Aucun'}
        """
        
        folium.Marker(
            location=[center_lat, center_long],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=colors[cluster_id % len(colors)], icon='info-sign')
        ).add_to(m)
    
    # Sauvegarder
    m.save(output_file)
    print(f"Carte sauvegardée: {output_file}")
    print(f"{'='*70}\n")


def analyze_cluster_content(df):
    """
    Analyse le contenu textuel de chaque cluster avec TF-IDF.
    
    Args:
        df: DataFrame avec 'cluster_hybrid' et tokens preprocessés
    """
    print(f"\n{'='*70}")
    print("ANALYSE DU CONTENU TEXTUEL PAR CLUSTER (TF-IDF)")
    print(f"{'='*70}")
    
    cluster_keywords = compute_tfidf_per_cluster(
        df,
        cluster_col='cluster_hybrid',
        text_cols=['tags', 'title'],
        top_n=10
    )
    
    display_cluster_keywords(
        cluster_keywords,
        title="Top 10 mots-clés par cluster (TF-IDF)"
    )


def visualize_spatial_clusters(df, output_file='../maps/clusters_spatial.html', sample_size=2000):
    """
    Visualise les clusters spatiaux purs sur une carte interactive de Lyon.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_spatial'
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher (pour la performance)
    """
    print(f"\n{'='*70}")
    print("VISUALISATION DES CLUSTERS SPATIAUX SUR CARTE")
    print(f"{'='*70}")
    
    # Calculer les mots-clés par cluster
    print("Calcul des mots-clés TF-IDF par cluster...")
    cluster_keywords = compute_tfidf_per_cluster(
        df,
        cluster_col='cluster_spatial',
        text_cols=['tags', 'title'],
        top_n=3
    )
    
    # Créer la carte centrée sur Lyon
    center_lat = df['lat'].mean()
    center_long = df['long'].mean()
    m = folium.Map(location=[center_lat, center_long], zoom_start=12)
    
    # Couleurs pour les clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'gray', 'black', 'lightgray', 'white', 'brown']
    
    # Échantillonner si trop de points
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    print(f"Affichage de {len(df_sample):,} points sur {len(df):,}")
    
    # Ajouter les points colorés par cluster
    for idx, row in df_sample.iterrows():
        cluster_id = int(row['cluster_spatial'])
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=colors[cluster_id % len(colors)],
            fill=True,
            fillColor=colors[cluster_id % len(colors)],
            fillOpacity=0.6,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)
    
    # Ajouter les centres des clusters
    for cluster_id in sorted(df['cluster_spatial'].unique()):
        cluster_df = df[df['cluster_spatial'] == cluster_id]
        center_lat = cluster_df['lat'].mean()
        center_long = cluster_df['long'].mean()
        
        # Récupérer les 3 mots-clés les plus pertinents
        keywords = cluster_keywords.get(cluster_id, [])
        keywords_text = '<br>'.join([f"{i+1}. {word}" for i, (word, score) in enumerate(keywords)])
        
        popup_html = f"""<b>Cluster {cluster_id}</b><br>
        Taille: {len(cluster_df)} photos<br>
        <br><b>Mots-clés:</b><br>
        {keywords_text if keywords_text else 'Aucun'}
        """
        
        folium.Marker(
            location=[center_lat, center_long],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=colors[cluster_id % len(colors)], icon='info-sign')
        ).add_to(m)
    
    # Sauvegarder
    m.save(output_file)
    print(f"Carte sauvegardée: {output_file}")
    print(f"{'='*70}\n")


def compare_spatial_vs_hybrid(df, n_clusters=10):
    """
    Compare le clustering purement spatial vs hybride.
    
    Args:
        df: DataFrame avec données
        n_clusters: nombre de clusters
    """
    print(f"\n{'='*70}")
    print("COMPARAISON: SPATIAL vs HYBRIDE")
    print(f"{'='*70}")
    
    # Clustering spatial pur
    print("\n1. Clustering SPATIAL (lat, long uniquement)...")
    kmeans_spatial = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df['cluster_spatial'] = kmeans_spatial.fit_predict(df[['lat', 'long']])
    silhouette_spatial = silhouette_score(df[['lat', 'long']], df['cluster_spatial'])
    
    # Visualiser le clustering spatial
    visualize_spatial_clusters(df, output_file='../maps/clusters_spatial.html')
    
    # Clustering hybride (déjà fait)
    if 'cluster_hybrid' in df.columns:
        print("\n2. Clustering HYBRIDE (déjà calculé)...")
        # Recalculer silhouette sur features spatiales uniquement pour comparaison
        silhouette_hybrid_spatial = silhouette_score(df[['lat', 'long']], df['cluster_hybrid'])
    else:
        print("\n2. Clustering HYBRIDE non disponible.")
        return
    
    print(f"\n{'='*70}")
    print("RÉSULTATS DE COMPARAISON")
    print(f"{'='*70}")
    print(f"{'Méthode':<20} {'Silhouette (spatial)':<25}")
    print('-'*70)
    print(f"{'Spatial pur':<20} {silhouette_spatial:<25.3f}")
    print(f"{'Hybride':<20} {silhouette_hybrid_spatial:<25.3f}")
    print(f"{'='*70}")
    
    # Statistiques de taille
    print("\nÉquilibre des tailles:")
    print(f"{'Méthode':<20} {'Min':<10} {'Max':<10} {'Moy':<10} {'Écart-type':<12}")
    print('-'*70)
    
    spatial_sizes = df['cluster_spatial'].value_counts()
    print(f"{'Spatial pur':<20} {spatial_sizes.min():<10} {spatial_sizes.max():<10} "
          f"{spatial_sizes.mean():<10.1f} {spatial_sizes.std():<12.1f}")
    
    hybrid_sizes = df['cluster_hybrid'].value_counts()
    print(f"{'Hybride':<20} {hybrid_sizes.min():<10} {hybrid_sizes.max():<10} "
          f"{hybrid_sizes.mean():<10.1f} {hybrid_sizes.std():<12.1f}")
    
    print(f"{'='*70}\n")


def main():
    """
    Fonction principale pour exécuter le clustering hybride.
    """
    print("\n" + "="*70)
    print(" "*15 + "CLUSTERING HYBRIDE - ZONES TOURISTIQUES LYON")
    print("="*70 + "\n")
    
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    print(f"Données chargées: {len(df):,} photos")
    
    # Limiter à un échantillon si trop de données
    sample_size = 5000
    if len(df) > sample_size:
        print(f"Échantillonnage de {sample_size:,} photos pour le clustering...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Paramètres du clustering
    N_CLUSTERS = 20
    SPATIAL_WEIGHT = 0.6
    TEXT_WEIGHT = 0.4
    
    # 1. Clustering hybride
    df, kmeans, vectorizer, feature_info = hybrid_clustering(
        df,
        n_clusters=N_CLUSTERS,
        spatial_weight=SPATIAL_WEIGHT,
        text_weight=TEXT_WEIGHT
    )
    
    # 2. Visualisation sur carte
    visualize_hybrid_clusters(df, output_file='../maps/clusters_hybrid.html')
    
    # 3. Analyse du contenu textuel
    analyze_cluster_content(df)
    
    # 4. Comparaison avec clustering spatial pur
    compare_spatial_vs_hybrid(df, n_clusters=N_CLUSTERS)
    
    # 5. Sauvegarder les résultats
    output_file = '../data/flickr_data2_hybrid_clustering.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Résultats sauvegardés: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
