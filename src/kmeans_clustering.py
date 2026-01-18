from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import os

from visualize_on_map import visualize_clusters_on_map
from cluster_evaluation import evaluate_clustering, plot_silhouette

def elbow_method(df, k_range=range(2, 21), show_plot=True):
    """
    Méthode du coude pour déterminer le nombre optimal de clusters.
    
    Args:
        df: DataFrame avec les colonnes 'lat' et 'long'
        k_range: Range de nombres de clusters à tester (par défaut 2 à 20)
        show_plot: Si True, affiche le graphique du coude
    
    Returns:
        dict: Dictionnaire avec 'k_values' et 'inertias'
    """
    # Sélection des coordonnées
    coords = df[['lat', 'long']]
    
    print(f"Calcul de l'inertie pour {len(k_range)} valeurs de k...")
    
    # Liste pour stocker les inertias
    inertias = []
    
    # Tester différentes valeurs de k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)
        print(f"  k={k:2d} → Inertie = {kmeans.inertia_:,.2f}")
    
    # Afficher le graphique du coude
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertias, 'bx-', linewidth=2, markersize=8)
    plt.xlabel('Nombre de clusters (k)', fontsize=12)
    plt.ylabel('Somme des distances au carré (Inertie)', fontsize=12)
    plt.title('Méthode du coude', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(k_range))
    plt.tight_layout()
    # Créer le dossier plots s'il n'existe pas
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/elbow_method.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graphique sauvegardé dans '../plots/elbow_method.png'")
    
    if show_plot:
        plt.show()
    
    return {
        'k_values': list(k_range),
        'inertias': inertias
    }

def run_kmeans_clustering(df, n_clusters=50):
    # 1. Sélection des colonnes géographiques
    # On travaille uniquement sur les positions GPS pour ce jalon
    coords = df[['lat', 'long']]
    
    # 2. Initialisation et entraînement de l'algorithme
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    
    # 3. Prédiction des clusters
    df['cluster_kmeans'] = kmeans.fit_predict(coords)
    
    print(f"Clustering terminé. {n_clusters} zones identifiées.")
    return df

def analyze_clusters(df):
    """
    Analyse les résultats du clustering.
    
    Args:
        df: DataFrame avec la colonne 'cluster_kmeans'
    """
    print("\n=== Analyse des clusters ===")
    
    # Nombre d'éléments par cluster
    cluster_counts = df['cluster_kmeans'].value_counts().sort_index()
    print("\nNombre de photos par cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count:,} photos")
    
    # Statistiques par cluster
    print("\nStatistiques par cluster:")
    cluster_stats = df.groupby('cluster_kmeans').agg({
        'id': 'count',
        'lat': 'mean',
        'long': 'mean'
    }).round(6)
    cluster_stats.columns = ['Nombre de photos', 'Latitude moyenne', 'Longitude moyenne']
    print(cluster_stats)
    
    return cluster_stats

# Exemple d'usage rapide :
if __name__ == "__main__":
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    
    # Déterminer le nombre optimal de clusters avec la méthode du coude
    print("=== Méthode du coude ===")
    elbow_results = elbow_method(df, k_range=range(2, 21), show_plot=False)
    
    # Utiliser un nombre de clusters choisi
    df_clustered = run_kmeans_clustering(df, n_clusters=9)
        
    # Visualiser sur une carte
    visualize_clusters_on_map(
        df_clustered,
        output_file='../maps/clusters_kmeans.html', 
        sample_size=2000, 
        cluster_col='cluster_kmeans',
        show_keywords=True
    )
    
    # Analyser les clusters
    cluster_stats = analyze_clusters(df_clustered)
    
    # Évaluation détaillée avec graphique de silhouette
    print("\n" + "="*70)
    print("ÉVALUATION DÉTAILLÉE DE LA QUALITÉ DU CLUSTERING")
    print("="*70)
    
    coords = df_clustered[['lat', 'long']]
    labels = df_clustered['cluster_kmeans'].values
    
    evaluation_results = evaluate_clustering(
        data=coords.values,
        labels=labels,
        metric='euclidean',
        method_name='K-Means (k=9)'
    )
    
    # Afficher le graphique de silhouette
    fig = plot_silhouette(
        sample_silhouette_values=evaluation_results['silhouette_samples'],
        silhouette_avg=evaluation_results['silhouette_avg'],
        labels=labels,
        n_clusters=evaluation_results['n_clusters'],
        title='Silhouette Plot - K-Means Clustering (k=9)',
        file_name='kmeans_silhouette.png',
        show_plot=True
    )

