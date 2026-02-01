import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

from text_mining_2 import preprocess_dataframe, compute_bm25_features, compute_bm25_per_cluster, display_cluster_keywords, BM25_AVAILABLE
from visualize_on_map import visualize_clusters_on_map
from cluster_evaluation import evaluate_clustering, plot_silhouette


def hdbscan_spatial_with_bm25_analysis(df, min_cluster_size=500, min_samples=50):
    """
    Nouvelle approche en 2 étapes :
    1. Clustering SPATIAL pur avec HDBSCAN (lat, long uniquement)
    2. Analyse textuelle BM25 sur chaque cluster spatial
    
    Cette approche est meilleure car :
    - Le clustering spatial détecte les zones géographiques
    - BM25 caractérise ensuite le contenu textuel de chaque zone
    - Pas de mélange artificiel spatial/textuel qui pollue les résultats
    
    Args:
        df: DataFrame avec colonnes lat, long, text_merged
        min_cluster_size: Taille minimale d'un cluster HDBSCAN
        min_samples: Nombre minimum d'échantillons dans un voisinage
    
    Returns:
        df: DataFrame avec colonne 'cluster_spatial_bm25'
        clusterer: Objet HDBSCAN entraîné
    """
    print("\n" + "="*70)
    print("CLUSTERING SPATIAL HDBSCAN + ANALYSE BM25")
    print("="*70 + "\n")
    print("Approche en 2 étapes :")
    print("  1. Clustering spatial pur (géographie)")
    print("  2. Analyse BM25 du contenu textuel par cluster")
    
    if 'text_merged' not in df.columns:
        print("\n⚠ Colonne text_merged absente, utilisation de texte vide")
        df['text_merged'] = ""
    
    print("\nPrétraitement des textes...")
    df = preprocess_dataframe(df, text_cols=['text_merged'])
    
    print("\nÉTAPE 1 : CLUSTERING SPATIAL HDBSCAN")
    print("-" * 70)
    
    coords_rad = np.radians(df[['lat', 'long']].values)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine',
        cluster_selection_method='eom',
        gen_min_span_tree=True
    )
    
    cluster_labels = clusterer.fit_predict(coords_rad)
    df['cluster_spatial_bm25'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nRésultats du clustering spatial :")
    print(f"  - Nombre de clusters géographiques : {n_clusters}")
    print(f"  - Points de bruit : {n_noise} ({n_noise/len(df)*100:.1f}%)")
    print(f"  - Points clusterisés : {len(df) - n_noise} ({(len(df)-n_noise)/len(df)*100:.1f}%)")
    
    print("\nÉTAPE 2 : ANALYSE BM25 DES CLUSTERS")
    print("-" * 70)
    print("Caractérisation textuelle de chaque zone géographique...")
    
    return df, clusterer


def main():
    """
    Fonction principale pour exécuter le clustering spatial HDBSCAN + analyse BM25.
    Approche en 2 étapes pour de meilleurs résultats.
    """
    print("\n" + "="*70)
    print(" "*8 + "CLUSTERING SPATIAL HDBSCAN + ANALYSE BM25 - LYON")
    print("="*70 + "\n")
    
    print("Chargement des données...")
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    print(f"Données chargées: {len(df):,} photos")
    
    sample_size = 135029
    if len(df) > sample_size:
        print(f"\nÉchantillonnage de {sample_size:,} photos pour le clustering...")
        df = df.sample(n=sample_size, random_state=42)
    else:
        print(f"\nUtilisation de toutes les données : {len(df):,} photos")
    
    print("\n" + "="*70)
    print("PARAMÈTRES DU CLUSTERING")
    print("="*70)
    print("Algorithme: HDBSCAN Spatial pur")
    print("Analyse textuelle: BM25 par cluster")
    print("min_cluster_size: 300 (réduit pour plus de clusters)")
    print("min_samples: 30 (réduit pour plus de sensibilité)")
    
    df, clusterer = hdbscan_spatial_with_bm25_analysis(
        df,
        min_cluster_size=300,
        min_samples=30
    )
    
    print("\nGénération de la carte...")
    visualize_clusters_on_map(
        df,
        output_file='../maps/clusters_spatial_bm25.html',
        show_keywords=True,
        cluster_col='cluster_spatial_bm25'
    )
    
    if BM25_AVAILABLE:
        print("\nAnalyse BM25 détaillée par cluster géographique...")
        cluster_keywords = compute_bm25_per_cluster(
            df,
            cluster_col='cluster_spatial_bm25',
            text_cols=['text_merged'],
            top_n=15
        )
        display_cluster_keywords(
            cluster_keywords,
            title="Mots-clés BM25 par zone géographique"
        )
    
    coords = df[['lat', 'long']].values
    labels = df['cluster_spatial_bm25'].values
    
    print("\nÉvaluation du clustering...")
    evaluation_results = evaluate_clustering(
        coords,
        labels,
        metric='euclidean',
        method_name='HDBSCAN Spatial + BM25 Analysis'
    )
    
    fig = plot_silhouette(
        sample_silhouette_values=evaluation_results['silhouette_samples'],
        silhouette_avg=evaluation_results['silhouette_avg'],
        labels=labels,
        n_clusters=evaluation_results['n_clusters'],
        title='Silhouette Plot - HDBSCAN Spatial + BM25 Analysis',
        file_name='plot_spatial_bm25_silhouette.png',
        show_plot=False
    )
    
    if clusterer.condensed_tree_ is not None:
        print("\nGénération du dendrogramme HDBSCAN...")
        os.makedirs('../plots', exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        clusterer.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=colors,
            axis=ax
        )
        
        ax.set_title('HDBSCAN Condensed Tree - Spatial + BM25 Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Number of points', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('../plots/plot_spatial_bm25_hierarchy.png', dpi=150, bbox_inches='tight')
        print("   Sauvegardé: ../plots/plot_spatial_bm25_hierarchy.png")
        plt.close()
    
    df.to_csv('../data/flickr_data2_spatial_bm25.csv', index=False)
    print(f"\n{'='*70}")
    print("Résultats sauvegardés:")
    print("  - ../data/flickr_data2_spatial_bm25.csv")
    print("  - ../maps/clusters_spatial_bm25.html")
    print("  - ../plots/plot_spatial_bm25_silhouette.png")
    print("  - ../plots/plot_spatial_bm25_hierarchy.png")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
