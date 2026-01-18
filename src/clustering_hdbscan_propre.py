"""
HDBSCAN Clustering pour données géolocalisées Flickr Lyon
Clustering basé sur la densité hiérarchique avec gestion automatique du bruit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import folium
import hdbscan

# Import des fonctions de preprocessing depuis text_mining
from text_mining import preprocess_dataframe, compute_tfidf_per_cluster, display_cluster_keywords

from visualize_on_map import visualize_clusters_on_map, analyze_cluster_content

from cluster_evaluation import evaluate_clustering, plot_silhouette

def hdbscan_spatial_only(df, min_cluster_size=50, min_samples=10):
    """
    Clustering HDBSCAN purement spatial optimisé pour données géographiques.
    
    Args:
        df: DataFrame avec lat, long
        min_cluster_size: taille minimale d'un cluster
        min_samples: nombre minimum d'échantillons
        
    Returns:
        df: DataFrame avec colonne 'cluster_spatial_hdbscan' ajoutée
    """
    print(f"\n{'='*70}")
    print(f"CLUSTERING HDBSCAN SPATIAL (lat/long uniquement)")
    print(f"{'='*70}")
    print(f"Paramètres:")
    print(f"  - Taille min cluster: {min_cluster_size}")
    print(f"  - Min samples: {min_samples}")
    print(f"  - Métrique: Haversine (distances géographiques réelles)")
    
    # Conversion en radians pour la métrique haversine
    coords_rad = np.radians(df[['lat', 'long']].values)
    
    # HDBSCAN avec métrique haversine pour données géographiques
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine',              # Métrique adaptée aux coordonnées GPS
        cluster_selection_method='eom',
        alpha=1.0,
        allow_single_cluster=False,
        gen_min_span_tree=True           # Générer le MST pour visualisation
    )
    
    cluster_labels = clusterer.fit_predict(coords_rad)
    df['cluster_spatial_hdbscan'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nRésultats:")
    print(f"  - Clusters trouvés: {n_clusters}")
    print(f"  - Points de bruit: {n_noise} ({n_noise/len(df)*100:.1f}%)")
    
    return df, clusterer

def plot_hdbscan_tree(clusterer, save_path='../plots/hdbscan_hierarchy.png'):
    """
    Visualise l'arbre hiérarchique de HDBSCAN (dendrogramme).
    Montre comment les clusters se forment à différentes échelles de densité.
    
    Args:
        clusterer: Modèle HDBSCAN entraîné (avec condensed_tree_)
        save_path: Chemin de sauvegarde du graphique
        
    Returns:
        fig: Figure matplotlib
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\n{'='*70}")
    print("VISUALISATION DE LA HIÉRARCHIE HDBSCAN (DENDROGRAMME)")
    print(f"{'='*70}")
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Générer une palette de couleurs discrètes (liste de couleurs)
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
    
    # Plot du dendrogramme condensé
    clusterer.condensed_tree_.plot(
        select_clusters=True,
        selection_palette=colors,
        axis=ax
    )
    
    ax.set_title('Arbre hiérarchique HDBSCAN - Zones touristiques Lyon', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Numéro de cluster / échantillon', fontsize=12)
    ax.set_ylabel('Distance (λ)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dendrogramme sauvegardé: {save_path}")
    
    # Informations sur la hiérarchie
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    print(f"\nInformations hiérarchiques:")
    print(f"  - Nombre de nœuds dans l'arbre: {len(clusterer.condensed_tree_._raw_tree)}")
    print(f"  - Nombre de clusters sélectionnés: {n_clusters}")
    print(f"  - Les branches longues = clusters stables et bien séparés")
    print(f"  - Les branches courtes = clusters fusionnés tôt, moins distincts")
    
    plt.show()
    
    return fig

def main():
    """
    Fonction principale pour exécuter le clustering HDBSCAN.
    """
    print("\n" + "="*70)
    print(" "*15 + "CLUSTERING HDBSCAN - ZONES TOURISTIQUES LYON")
    print("="*70 + "\n")
    
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    print(f"Données chargées: {len(df):,} photos")
    
    # Limiter à un échantillon si trop de données
    sample_size = 10000000
    if len(df) > sample_size:
        print(f"Échantillonnage de {sample_size:,} photos pour le clustering...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Paramètres du clustering (optimisés pour Lyon)
    MIN_CLUSTER_SIZE = 500   # Taille min pour un cluster (réduit pour détecter plus de zones)
    MIN_SAMPLES = 50        # Échantillons min dans voisinage (réduit pour plus de sensibilité)
    
    # Clustering spatial pur
    df, clusterer = hdbscan_spatial_only(df, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
    
    # Visualisation du clustering spatial 
    print("\nGénération de la carte pour clustering spatial...")
    # Créer une copie temporaire pour la visualisation
    df_temp = df.copy()

    visualize_clusters_on_map(
        df_temp,
        output_file='../maps/clusters_hdbscan_spatial_propre.html',
        show_keywords=True,
        cluster_col='cluster_spatial_hdbscan',
    )
    
    # Visualisations hiérarchiques
    # Dendrogramme (arbre de clustering)
    plot_hdbscan_tree(clusterer, save_path='../plots/hdbscan_hierarchy.png')

    # Évaluation détaillée avec graphique de silhouette
    print("\n" + "="*70)
    print("ÉVALUATION DÉTAILLÉE DE LA QUALITÉ DU CLUSTERING")
    print("="*70)
    
    coords = df_temp[['lat', 'long']]
    labels = df_temp['cluster_label'].values
    
    evaluation_results = evaluate_clustering(
        data=coords.values,
        labels=labels,
        metric='euclidean',
        method_name='HDBSCAN Spatial'
    )
    
    # Afficher le graphique de silhouette
    fig = plot_silhouette(
        sample_silhouette_values=evaluation_results['silhouette_samples'],
        silhouette_avg=evaluation_results['silhouette_avg'],
        labels=labels,
        n_clusters=evaluation_results['n_clusters'],
        title='Silhouette Plot - HDBSCAN Spatial',
        file_name='hdbscan_spatial_propre_silhouette.png',
        show_plot=True
    )


if __name__ == '__main__':
    main()
