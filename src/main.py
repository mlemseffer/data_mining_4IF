"""
Fichier principal pour l'analyse de clustering et text mining des photos de Lyon.
Interface interactive pour choisir une méthode de clustering et visualiser les résultats.
"""

import pandas as pd
from text_mining import preprocess_dataframe, compute_tfidf_per_cluster, display_cluster_keywords
from hierarchical_clustering import run_hierarchical_clustering, visualize_clusters_on_map
from hybrid_clustering import hybrid_clustering, visualize_hybrid_clusters, analyze_cluster_content


def print_statistics(df, cluster_col='cluster'):
    """Affiche les statistiques détaillées par cluster."""
    print(f"\n{'='*70}")
    print("STATISTIQUES PAR CLUSTER")
    print(f"{'='*70}")
    print(f"{'Cluster':<10} {'Taille':<10} {'Lat moy':<12} {'Long moy':<12} {'% du total':<12}")
    print('-'*70)
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id]
        size = len(cluster_df)
        lat_mean = cluster_df['lat'].mean()
        long_mean = cluster_df['long'].mean()
        percentage = (size / len(df)) * 100
        print(f"{cluster_id:<10} {size:<10} {lat_mean:<12.4f} {long_mean:<12.4f} {percentage:<12.1f}%")
    
    print('-'*70)
    cluster_sizes = df[cluster_col].value_counts()
    print(f"Taille min: {cluster_sizes.min()}")
    print(f"Taille max: {cluster_sizes.max()}")
    print(f"Taille moyenne: {cluster_sizes.mean():.1f}")
    print(f"Écart-type: {cluster_sizes.std():.1f}")
    print(f"{'='*70}")


def analyze_text_mining(df, cluster_col='cluster'):
    """Analyse le contenu textuel avec TF-IDF."""
    print(f"\n{'='*70}")
    print("ANALYSE DU CONTENU TEXTUEL PAR CLUSTER (TF-IDF)")
    print(f"{'='*70}")
    
    cluster_keywords = compute_tfidf_per_cluster(
        df,
        cluster_col=cluster_col,
        text_cols=['tags', 'title'],
        top_n=10
    )
    
    display_cluster_keywords(cluster_keywords, title="Top 10 mots-clés par cluster")


def main():
    """
    Fonction principale interactive pour l'analyse de clustering.
    """
    print("\n" + "="*70)
    print(" "*15 + "ANALYSE DE CLUSTERING - ZONES TOURISTIQUES LYON")
    print("="*70 + "\n")
    
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    print(f"Données chargées: {len(df):,} photos")
    
    # Limiter à un échantillon si nécessaire
    sample_size = 5000
    if len(df) > sample_size:
        print(f"\nÉchantillonnage de {sample_size:,} photos pour l'analyse...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Prétraiter les textes
    print("\nPrétraitement des textes (tags et title)...")
    df = preprocess_dataframe(df, text_cols=['tags', 'title'])
    
    # Choix de la méthode de clustering
    print("\n" + "="*70)
    print("CHOIX DE LA MÉTHODE DE CLUSTERING")
    print("="*70)
    print("1. Clustering Hiérarchique - Complete")
    print("2. Clustering Hiérarchique - Average")
    print("3. Clustering Hiérarchique - Single")
    print("4. Clustering Hybride (Spatial + Textuel)")
    
    choice = input("\nChoisissez une méthode (1-4): ").strip()
    
    # Demander le nombre de clusters
    n_clusters_input = input("Nombre de clusters souhaités (défaut: 10): ").strip()
    n_clusters = int(n_clusters_input) if n_clusters_input else 10
    
    # Exécuter la méthode choisie
    if choice == '1':
        df, model, silhouette = run_hierarchical_clustering(
            df, 
            n_clusters=n_clusters, 
            linkage='complete',
            show_dendrogram=False
        )
        print_statistics(df, 'cluster')
        analyze_text_mining(df, 'cluster')
        visualize_clusters_on_map(
            df, 
            cluster_col='cluster',
            output_file='../maps/clusters_hierarchical_complete.html'
        )
        output_file = '../data/flickr_data2_hierarchical_complete.csv'
        
    elif choice == '2':
        df, model, silhouette = run_hierarchical_clustering(
            df, 
            n_clusters=n_clusters, 
            linkage='average',
            show_dendrogram=False
        )
        print_statistics(df, 'cluster')
        analyze_text_mining(df, 'cluster')
        visualize_clusters_on_map(
            df, 
            cluster_col='cluster',
            output_file='../maps/clusters_hierarchical_average.html'
        )
        output_file = '../data/flickr_data2_hierarchical_average.csv'
        
    elif choice == '3':
        df, model, silhouette = run_hierarchical_clustering(
            df, 
            n_clusters=n_clusters, 
            linkage='single',
            show_dendrogram=False
        )
        print_statistics(df, 'cluster')
        analyze_text_mining(df, 'cluster')
        visualize_clusters_on_map(
            df, 
            cluster_col='cluster',
            output_file='../maps/clusters_hierarchical_single.html'
        )
        output_file = '../data/flickr_data2_hierarchical_single.csv'
        
    elif choice == '4':
        print("\nParamètres du clustering hybride:")
        spatial_weight = input("Poids spatial (0-1, défaut: 0.7): ").strip()
        spatial_weight = float(spatial_weight) if spatial_weight else 0.7
        text_weight = 1 - spatial_weight
        
        df, kmeans, vectorizer, feature_info = hybrid_clustering(
            df,
            n_clusters=n_clusters,
            spatial_weight=spatial_weight,
            text_weight=text_weight
        )
        visualize_hybrid_clusters(df, output_file='../maps/clusters_hybrid.html')
        analyze_cluster_content(df)
        output_file = '../data/flickr_data2_hybrid.csv'
        
    else:
        print("Choix invalide. Utilisation du clustering hiérarchique complete par défaut.")
        df, model, silhouette = run_hierarchical_clustering(
            df, 
            n_clusters=n_clusters, 
            linkage='complete',
            show_dendrogram=False
        )
        print_statistics(df, 'cluster')
        analyze_text_mining(df, 'cluster')
        visualize_clusters_on_map(
            df, 
            cluster_col='cluster',
            output_file='../maps/clusters_hierarchical_complete.html'
        )
        output_file = '../data/flickr_data2_hierarchical_complete.csv'
    
    # Sauvegarder les résultats
    df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Résultats sauvegardés: {output_file}")
    print(f"{'='*70}\n")
    
    print("Analyse terminée avec succès!")


if __name__ == '__main__':
    main()
