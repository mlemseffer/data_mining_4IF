"""
Fichier principal pour l'analyse de clustering et text mining des photos de Lyon.
Interface interactive pour choisir une méthode de clustering et visualiser les résultats.
"""

import pandas as pd
from text_mining import preprocess_dataframe, compute_tfidf_per_cluster, display_cluster_keywords
from hierarchical_clustering import run_hierarchical_clustering
from visualize_on_map import visualize_clusters_on_map
from hybrid_clustering import hybrid_clustering, analyze_cluster_content


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
        text_cols=['text_merged'],
        top_n=10
    )
    
    display_cluster_keywords(cluster_keywords, title="Top 10 mots-clés par cluster")


def run_all_algorithms(df, n_clusters=10):
    """
    Exécute tous les algorithmes de clustering disponibles.
    
    Args:
        df: DataFrame avec les données
        n_clusters: nombre de clusters souhaité
        
    Returns:
        dict: résultats de tous les algorithmes
    """
    results = {}
    
    print("\n" + "="*70)
    print(" "*10 + "EXÉCUTION DE TOUS LES ALGORITHMES DE CLUSTERING")
    print("="*70)
    
    # 1. Clustering Hiérarchique - Complete
    print("\n" + ">"*35)
    print("1/8 - CLUSTERING HIÉRARCHIQUE (COMPLETE)")
    print(">"*35)
    df_temp = df.copy()
    df_temp, model, silhouette = run_hierarchical_clustering(
        df_temp, 
        n_clusters=n_clusters, 
        linkage='complete',
        show_dendrogram=False
    )
    print_statistics(df_temp, 'cluster')
    analyze_text_mining(df_temp, 'cluster')
    df_temp['cluster_label'] = df_temp['cluster']
    visualize_clusters_on_map(
        df_temp, 
        output_file='../maps/clusters_hierarchical_complete.html',
        show_keywords=False
    )
    df_temp.to_csv('../data/flickr_data2_hierarchical_complete.csv', index=False)
    results['hierarchical_complete'] = {'silhouette': silhouette, 'n_clusters': n_clusters}
    print("[OK] Clustering Hiérarchique Complete terminé")
    
    # 2. Clustering Hiérarchique - Average
    print("\n" + ">"*35)
    print("2/8 - CLUSTERING HIÉRARCHIQUE (AVERAGE)")
    print(">"*35)
    df_temp = df.copy()
    df_temp, model, silhouette = run_hierarchical_clustering(
        df_temp, 
        n_clusters=n_clusters, 
        linkage='average',
        show_dendrogram=False
    )
    print_statistics(df_temp, 'cluster')
    analyze_text_mining(df_temp, 'cluster')
    df_temp['cluster_label'] = df_temp['cluster']
    visualize_clusters_on_map(
        df_temp, 
        output_file='../maps/clusters_hierarchical_average.html',
        show_keywords=False
    )
    df_temp.to_csv('../data/flickr_data2_hierarchical_average.csv', index=False)
    results['hierarchical_average'] = {'silhouette': silhouette, 'n_clusters': n_clusters}
    print("[OK] Clustering Hiérarchique Average terminé")
    
    # 3. Clustering Hiérarchique - Single
    print("\n" + ">"*35)
    print("3/8 - CLUSTERING HIÉRARCHIQUE (SINGLE)")
    print(">"*35)
    df_temp = df.copy()
    df_temp, model, silhouette = run_hierarchical_clustering(
        df_temp, 
        n_clusters=n_clusters, 
        linkage='single',
        show_dendrogram=False
    )
    print_statistics(df_temp, 'cluster')
    analyze_text_mining(df_temp, 'cluster')
    df_temp['cluster_label'] = df_temp['cluster']
    visualize_clusters_on_map(
        df_temp, 
        output_file='../maps/clusters_hierarchical_single.html',
        show_keywords=False
    )
    df_temp.to_csv('../data/flickr_data2_hierarchical_single.csv', index=False)
    results['hierarchical_single'] = {'silhouette': silhouette, 'n_clusters': n_clusters}
    print("[OK] Clustering Hiérarchique Single terminé")
    
    # 4. Clustering Hybride
    print("\n" + ">"*35)
    print("4/8 - CLUSTERING HYBRIDE (SPATIAL + TEXTUEL)")
    print(">"*35)
    df_temp = df.copy()
    df_temp, kmeans, vectorizer, feature_info = hybrid_clustering(
        df_temp,
        n_clusters=n_clusters,
        spatial_weight=0.7,
        text_weight=0.3
    )
    df_temp['cluster_label'] = df_temp['cluster_hybrid']
    visualize_clusters_on_map(
        df_temp, 
        output_file='../maps/clusters_hybrid.html',
        show_keywords=True
    )
    analyze_cluster_content(df_temp)
    df_temp.to_csv('../data/flickr_data2_hybrid.csv', index=False)
    results['hybrid'] = {'n_clusters': n_clusters}
    print("[OK] Clustering Hybride terminé")
    
    # 5. K-Means
    try:
        from kmeans_clustering import run_kmeans_clustering
        print("\n" + ">"*35)
        print("5/8 - CLUSTERING K-MEANS")
        print(">"*35)
        df_temp = df.copy()
        df_temp = run_kmeans_clustering(df_temp, n_clusters=n_clusters)
        df_temp['cluster_label'] = df_temp['cluster_kmeans']
        visualize_clusters_on_map(
            df_temp,
            output_file='../maps/clusters_kmeans.html',
            show_keywords=False
        )
        df_temp.to_csv('../data/flickr_data2_kmeans.csv', index=False)
        results['kmeans'] = {'n_clusters': n_clusters}
        print("[OK] Clustering K-Means terminé")
    except ImportError:
        print("[WARN] K-Means non disponible")
    
    # 6. BIRCH
    try:
        from birch_clustering import birch_clustering
        from sklearn.metrics import silhouette_score
        print("\n" + ">"*35)
        print("6/8 - CLUSTERING BIRCH")
        print(">"*35)
        df_temp = df.copy()
        X = df_temp[['lat', 'long']].values
        model = birch_clustering(X, n_clusters=n_clusters)
        df_temp['birch_cluster'] = model.labels_
        silhouette = silhouette_score(X, model.labels_, metric='euclidean')
        print(f"Score de silhouette: {silhouette:.3f}")
        df_temp['cluster_label'] = df_temp['birch_cluster']
        visualize_clusters_on_map(
            df_temp,
            output_file='../maps/clusters_birch.html',
            show_keywords=False
        )
        df_temp.to_csv('../data/flickr_data2_birch.csv', index=False)
        results['birch'] = {'silhouette': silhouette, 'n_clusters': n_clusters}
        print("[OK] Clustering BIRCH terminé")
    except ImportError:
        print("[WARN] BIRCH non disponible")
    
    # 7. DBSCAN
    try:
        from dbscan import run_dbscan, find_optimal_eps, analyze_dbscan_clusters
        print("\n" + ">"*35)
        print("7/8 - CLUSTERING DBSCAN")
        print(">"*35)
        df_temp = df.copy()
        suggested_eps = find_optimal_eps(df_temp, min_samples=15)
        df_temp = run_dbscan(df_temp, eps=suggested_eps, min_samples=15)
        analyze_dbscan_clusters(df_temp)
        visualize_clusters_on_map(
            df_temp, 
            output_file='../maps/clusters_dbscan.html',
            show_keywords=False
        )
        df_temp.to_csv('../data/flickr_data2_dbscan.csv', index=False)
        n_clusters_dbscan = len(set(df_temp['cluster_label'])) - (1 if -1 in df_temp['cluster_label'].values else 0)
        results['dbscan'] = {'n_clusters': n_clusters_dbscan}
        print("[OK] Clustering DBSCAN terminé")
    except ImportError:
        print("[WARN] DBSCAN non disponible")
    
    # 8. HDBSCAN
    try:
        from hdbscan_clustering import hdbscan_clustering as hdbscan_clust, analyze_cluster_content as hdbscan_analyze
        print("\n" + ">"*35)
        print("8/8 - CLUSTERING HDBSCAN")
        print(">"*35)
        df_temp = df.copy()
        df_temp, clusterer, vectorizer, feature_info = hdbscan_clust(
            df_temp,
            min_cluster_size=500,
            min_samples=50,
            spatial_weight=0.7,
            text_weight=0.3
        )
        visualize_clusters_on_map(
            df_temp, 
            output_file='../maps/clusters_hdbscan.html',
            show_keywords=True
        )
        hdbscan_analyze(df_temp)
        df_temp.to_csv('../data/flickr_data2_hdbscan.csv', index=False)
        n_clusters_hdbscan = len(set(df_temp['cluster_label'])) - (1 if -1 in df_temp['cluster_label'].values else 0)
        results['hdbscan'] = {'n_clusters': n_clusters_hdbscan}
        print("[OK] Clustering HDBSCAN terminé")
    except ImportError as e:
        print(f"[WARN] HDBSCAN non disponible: {e}")
    
    # Résumé final
    print("\n" + "="*70)
    print(" "*20 + "RÉSUMÉ DES RÉSULTATS")
    print("="*70)
    print(f"\n{'Algorithme':<30} {'Clusters':<15} {'Silhouette':<15}")
    print("-"*70)
    for algo, res in results.items():
        silhouette_str = f"{res['silhouette']:.3f}" if 'silhouette' in res else "N/A"
        print(f"{algo:<30} {res.get('n_clusters', 'N/A'):<15} {silhouette_str:<15}")
    print("="*70)
    
    print("\n[OK] Tous les algorithmes ont été exécutés avec succès!")
    print(f"[INFO] Cartes HTML sauvegardées dans: ../maps/")
    print(f"[INFO] Données sauvegardées dans: ../data/")
    
    return results


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
    print("\nPrétraitement des textes (text_merged)...")
    df = preprocess_dataframe(df, text_cols=['text_merged'])
    
    # Choix de la méthode de clustering
    print("\n" + "="*70)
    print("CHOIX DE LA MÉTHODE DE CLUSTERING")
    print("="*70)
    print("1. Clustering Hiérarchique - Complete")
    print("2. Clustering Hiérarchique - Average")
    print("3. Clustering Hiérarchique - Single")
    print("4. Clustering Hybride (Spatial + Textuel)")
    print("5. Exécuter TOUS les algorithmes (échantillon de 5000)")
    print("6. Exécuter TOUS les algorithmes (échantillon large de 20000)")
    
    choice = input("\nChoisissez une méthode (1-6): ").strip()
    
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
        df['cluster_label'] = df['cluster_hybrid']
        visualize_clusters_on_map(
            df, 
            output_file='../maps/clusters_hybrid.html',
            show_keywords=True
        )
        analyze_cluster_content(df)
        output_file = '../data/flickr_data2_hybrid.csv'
        
    elif choice == '5':
        # Exécuter tous les algorithmes sur un échantillon de 5000
        results = run_all_algorithms(df, n_clusters=n_clusters)
        return  # Terminer après avoir tout exécuté
        
    elif choice == '6':
        # Exécuter tous les algorithmes sur un échantillon large de 20000
        print("\n" + "!"*70)
        print("ATTENTION: Traitement d'un large échantillon (20k photos)")
        print("Cela peut prendre 30-60 minutes selon votre machine!")
        print("!"*70)
        confirm = input("\nÊtes-vous sûr de vouloir continuer? (oui/non): ").strip().lower()
        if confirm in ['oui', 'o', 'yes', 'y']:
            # Recharger avec échantillon de 20k
            df_large = pd.read_csv('../data/flickr_data2_cleaned.csv')
            sample_size = 20000
            if len(df_large) > sample_size:
                print(f"\n[OK] Échantillonnage de {sample_size:,} photos sur {len(df_large):,}")
                df_large = df_large.sample(n=sample_size, random_state=42)
            df_large = preprocess_dataframe(df_large, text_cols=['text_merged'])
            results = run_all_algorithms(df_large, n_clusters=n_clusters)
        else:
            print("[ANNULE] Opération annulée.")
        return
        
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
