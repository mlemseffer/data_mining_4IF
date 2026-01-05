from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import os

def elbow_method(df, k_range=range(2, 21)):
    """
    Méthode du coude pour déterminer le nombre optimal de clusters.
    
    Args:
        df: DataFrame avec les colonnes 'lat' et 'long'
        k_range: Range de nombres de clusters à tester (par défaut 2 à 20)
    
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
    plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graphique sauvegardé dans 'elbow_method.png'")
    plt.show()
    
    return {
        'k_values': list(k_range),
        'inertias': inertias
    }

def run_first_clustering(df, n_clusters=50):
    # 1. Sélection des colonnes géographiques
    # On travaille uniquement sur les positions GPS pour ce jalon
    coords = df[['lat', 'long']]
    
    # 2. Initialisation et entraînement de l'algorithme
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    
    # 3. Prédiction des clusters
    df['cluster_label'] = kmeans.fit_predict(coords)
    
    print(f"Clustering terminé. {n_clusters} zones identifiées.")
    return df

def analyze_clusters(df):
    """
    Analyse les résultats du clustering.
    
    Args:
        df: DataFrame avec la colonne 'cluster_label'
    """
    print("\n=== Analyse des clusters ===")
    
    # Nombre d'éléments par cluster
    cluster_counts = df['cluster_label'].value_counts().sort_index()
    print("\nNombre de photos par cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count:,} photos")
    
    # Statistiques par cluster
    print("\nStatistiques par cluster:")
    cluster_stats = df.groupby('cluster_label').agg({
        'id': 'count',
        'lat': 'mean',
        'long': 'mean'
    }).round(6)
    cluster_stats.columns = ['Nombre de photos', 'Latitude moyenne', 'Longitude moyenne']
    print(cluster_stats)
    
    return cluster_stats

def evaluate_clustering_quality(df):
    """
    Évalue la qualité du clustering avec le score de silhouette.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_label'
    
    Returns:
        float: Score de silhouette moyen
    """
    from sklearn.metrics import silhouette_score
    
    coords = df[['lat', 'long']]
    labels = df['cluster_label']
    
    silhouette_avg = silhouette_score(coords, labels, metric='euclidean')
    # Scores récupérés du premier TP
    print(f"\n=== Qualité du clustering ===")
    print(f"Score de silhouette moyen: {silhouette_avg:.4f}")
    print("  > 0.7  : Clustering excellent")
    print("  0.5-0.7: Clustering bon")
    print("  0.25-0.5: Clustering moyen")
    print("  < 0.25 : Clustering faible")
    
    return silhouette_avg

def ensure_directory_exists(directory):
    """
    Vérifie si un dossier existe, sinon le crée.
    
    Args:
        directory (str): Chemin du dossier à vérifier/créer.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_clusters_on_map(df, output_file='clusters_map.html', sample_size=1000):
    """
    Visualise les clusters sur une carte interactive de Lyon.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_label'
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher (pour la performance)
    """
    import folium
    
    print(f"\n=== Visualisation des clusters ===")
    
    # Créer le dossier pour la carte si nécessaire
    ensure_directory_exists(os.path.dirname(output_file))
    
    # Créer une carte centrée sur Lyon
    m = folium.Map(location=[45.75, 4.85], zoom_start=12)
    
    # Couleurs pour les clusters (jusqu'à 20 clusters)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'gray', 'black', 'lightgray', 'white', 'brown']
    
    # Échantillonner pour la performance si nécessaire
    df_sample = df.sample(min(sample_size, len(df)))
    
    # Ajouter les points avec leur couleur de cluster
    for idx, row in df_sample.iterrows():
        cluster_id = row['cluster_label']
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=colors[cluster_id % len(colors)],
            fill=True,
            fillColor=colors[cluster_id % len(colors)],
            fillOpacity=0.6,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)
    
    # Sauvegarder la carte
    m.save(output_file)
    print(f"Carte sauvegardée dans '{output_file}'")
    print(f"{len(df_sample):,} points affichés sur {len(df):,} total")

# Exemple d'usage rapide :
if __name__ == "__main__":
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    
    # Déterminer le nombre optimal de clusters avec la méthode du coude
    print("=== Méthode du coude ===")
#    elbow_results = elbow_method(df, k_range=range(2, 21))
    
    # Utiliser un nombre de clusters choisi
    df_clustered = run_first_clustering(df, n_clusters=9)
    
    # Analyser les clusters
    cluster_stats = analyze_clusters(df_clustered)
    
    # Évaluer la qualité
    silhouette = evaluate_clustering_quality(df_clustered)
    
    # Visualiser sur une carte
    visualize_clusters_on_map(df_clustered, output_file='../maps/clusters_lyon.html', sample_size=2000)
