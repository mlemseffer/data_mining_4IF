from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_eps(df, min_samples=50):
    """
    Trouve la valeur optimale d'eps en utilisant la méthode du k-distance.
    Affiche le graphique en mètres pour une interprétation facile.
    
    Args:
        df: DataFrame avec les colonnes 'lat' et 'long'
        min_samples: Nombre minimum de points (k-voisin)
    
    Returns:
        float: Valeur suggérée pour eps (en RADIANS)
    """
    # 1. Préparation des coordonnées en RADIANS
    coords = df[['lat', 'long']].values
    coords_rad = np.radians(coords)
    
    print(f"=== Recherche de eps optimal ===")
    print(f"Nombre de points analysés: {len(coords)}")
    print(f"k (min_samples): {min_samples}")
    
    # 2. Calcul des plus proches voisins avec la métrique Haversine
    # La distance retournée par haversine est en radians
    neigh = NearestNeighbors(n_neighbors=min_samples, metric='haversine')
    neigh.fit(coords_rad)
    distances, _ = neigh.kneighbors(coords_rad)
    
    # On prend la distance au k-ième voisin (le plus éloigné du groupe)
    k_distances_rad = np.sort(distances[:, min_samples-1])
    
    # 3. Conversion en MÈTRES pour la visualisation
    # Rayon de la Terre = 6 371 000 mètres
    k_distances_meters = k_distances_rad * 6371000
    
    # 4. Création du graphique k-distance
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances_meters)), k_distances_meters, 'b-', linewidth=2)
    plt.axhline(y=np.percentile(k_distances_meters, 90), color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Points (triés par distance)', fontsize=12)
    plt.ylabel(f'Distance au {min_samples}ème voisin (Mètres)', fontsize=12)
    plt.title(f'K-distance Graph (Lyon) - k={min_samples}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Zoomer sur la partie utile si nécessaire (on ignore les outliers extrêmes)
    plt.ylim(0, np.percentile(k_distances_meters, 98)) 
    
    plt.tight_layout()
    plt.savefig('k_distance_graph.png', dpi=150)
    print("\n📊 Graphique sauvegardé : 'k_distance_graph.png'")
    plt.show()
    
    # 5. Aide au diagnostic
    print(f"\n📊 Analyse des distances (en MÈTRES) :")
    percentiles = [50, 75, 85, 90, 95, 98]
    for p in percentiles:
        val_m = np.percentile(k_distances_meters, p)
        print(f"   Percentile {p:2d}%: {val_m:.2f} mètres")
    
    # 6. Suggestion
    # Le percentile 90 est souvent proche du "coude"
    suggested_eps_rad = np.percentile(k_distances_rad, 75)
    suggested_eps_m = suggested_eps_rad * 6371000
    
    print(f"\n💡 Valeur suggérée au percentile 75 : {suggested_eps_m:.1f} mètres")
    print(f"   Soit eps = {suggested_eps_rad:.8f} radians")
    print(f"   Si le cluster est trop grand (toute la ville) -> Baissez cette valeur (ex: 100m)")
    print(f"   Si trop de bruit (points isolés) -> Augmentez min_samples")
    
    return suggested_eps_rad

def test_dbscan_parameters(df, eps_range, min_samples_range):
    """
    Teste différentes combinaisons de paramètres eps et min_samples.
    
    Args:
        df: DataFrame avec les colonnes 'lat' et 'long'
        eps_range: Liste de valeurs eps à tester
        min_samples_range: Liste de valeurs min_samples à tester
    
    Returns:
        DataFrame: Résultats des tests avec scores
    """
    coords = df[['lat', 'long']].values
    
    print(f"\n=== Test des paramètres DBSCAN ===")
    print(f"Nombre de combinaisons à tester: {len(eps_range) * len(min_samples_range)}")
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            # Apply DBSCAN
            coords_rad = np.radians(df[['lat', 'long']].values)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
            labels = dbscan.fit_predict(coords_rad)
            
            # Compter les clusters et le bruit
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculer le score de silhouette (si au moins 2 clusters)
            if n_clusters >= 2:
                # Exclure le bruit pour le calcul de silhouette
                mask = labels != -1
                if mask.sum() > 0:
                    try:
                        silhouette = silhouette_score(coords[mask], labels[mask], metric='haversine')
                    except:
                        silhouette = -1
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percent': (n_noise / len(labels)) * 100,
                'silhouette': silhouette
            })
            
            print(f"  eps={eps:.6f}, min_samples={min_samples:2d} → "
                  f"clusters={n_clusters:3d}, bruit={n_noise:5d} ({n_noise/len(labels)*100:5.2f}%), "
                  f"silhouette={silhouette:6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Trouver la meilleure combinaison
    valid_results = results_df[results_df['silhouette'] > 0]
    if len(valid_results) > 0:
        best_idx = valid_results['silhouette'].idxmax()
        best = results_df.loc[best_idx]
        print(f"\n✅ Meilleure combinaison selon silhouette:")
        print(f"   eps={best['eps']:.6f}, min_samples={best['min_samples']:.0f}")
        print(f"   → {best['n_clusters']:.0f} clusters, {best['noise_percent']:.2f}% bruit, silhouette={best['silhouette']:.3f}")
    
    return results_df

def run_dbscan(df, eps, min_samples=4, metric='haversine'):
    """
    Applique DBSCAN avec les paramètres donnés.
    
    Args:
        df: DataFrame avec les colonnes 'lat' et 'long'
        eps: Rayon maximum de voisinage (en radians si metric='haversine', sinon en degrés)
        min_samples: Nombre minimum de points pour former un cluster
        metric: Métrique de distance ('haversine' recommandé pour GPS, ou 'euclidean')
    
    Returns:
        DataFrame: DataFrame avec la colonne 'cluster_label' ajoutée
    """
    coords = df[['lat', 'long']].values
    
    print(f"\n=== Application de DBSCAN ===")
    
    if metric == 'haversine':
        # Convertir en radians pour haversine
        coords_rad = np.radians(coords)
        eps_km = eps * 6371  # Conversion pour affichage
        print(f"eps={eps:.6f} radians (≈ {eps_km:.3f} km), min_samples={min_samples}")
        print(f"Métrique: haversine (distance géodésique)")
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
        df['cluster_label'] = dbscan.fit_predict(coords_rad)
    else:
        print(f"eps={eps:.6f} degrés, min_samples={min_samples}")
        print(f"Métrique: {metric}")
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        df['cluster_label'] = dbscan.fit_predict(coords)
    
    # Statistiques
    n_clusters = len(set(df['cluster_label'])) - (1 if -1 in df['cluster_label'].values else 0)
    n_noise = (df['cluster_label'] == -1).sum()
    
    print(f"✅ Clustering terminé:")
    print(f"   Nombre de clusters: {n_clusters}")
    print(f"   Points de bruit: {n_noise} ({n_noise/len(df)*100:.2f}%)")
    print(f"   Points dans des clusters: {len(df) - n_noise} ({(len(df)-n_noise)/len(df)*100:.2f}%)")
    
    # Avertissement si trop de clusters
    if n_clusters > 100:
        print(f"\n⚠️  ATTENTION: Nombre de clusters très élevé ({n_clusters})!")
        print(f"   → Augmentez eps pour avoir moins de clusters")
        print(f"   → Essayez par exemple: eps * 2 ou eps * 5")
    elif n_clusters < 3:
        print(f"\n⚠️  ATTENTION: Très peu de clusters ({n_clusters})")
        print(f"   → Diminuez eps pour avoir plus de clusters")
        print(f"   → Essayez par exemple: eps / 2 ou eps / 5")
    
    return df

def analyze_dbscan_clusters(df):
    """
    Analyse les résultats du clustering DBSCAN.
    
    Args:
        df: DataFrame avec la colonne 'cluster_label'
    """
    print("\n=== Analyse des clusters DBSCAN ===")
    
    # Séparer le bruit des clusters
    noise = df[df['cluster_label'] == -1]
    clustered = df[df['cluster_label'] != -1]
    
    print(f"\nPoints de bruit: {len(noise)} ({len(noise)/len(df)*100:.2f}%)")
    print(f"Points clusterisés: {len(clustered)} ({len(clustered)/len(df)*100:.2f}%)")
    
    if len(clustered) > 0:
        # Nombre de photos par cluster
        cluster_counts = clustered['cluster_label'].value_counts().sort_index()
        print(f"\nNombre de clusters: {len(cluster_counts)}")
        print("\nNombre de photos par cluster:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count:,} photos")
        
        # Statistiques par cluster
        print("\nStatistiques par cluster:")
        cluster_stats = clustered.groupby('cluster_label').agg({
            'id': 'count',
            'lat': ['mean', 'std'],
            'long': ['mean', 'std']
        }).round(6)
        cluster_stats.columns = ['Nombre', 'Lat_mean', 'Lat_std', 'Long_mean', 'Long_std']
        print(cluster_stats)
        
        return cluster_stats
    else:
        print("\nAucun cluster trouvé!")
        return None

def visualize_dbscan_on_map(df, output_file='dbscan_clusters_map.html', sample_size=2000):
    """
    Visualise les clusters DBSCAN sur une carte interactive.
    Le bruit est affiché en gris.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_label'
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher
    """
    import folium
    
    print(f"\n=== Visualisation des clusters DBSCAN ===")
    
    # Créer une carte centrée sur Lyon
    m = folium.Map(location=[45.75, 4.85], zoom_start=12)
    
    # Couleurs pour les clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'brown', 'black', 'white']
    
    # Échantillonner
    df_sample = df.sample(min(sample_size, len(df)))
    
    # Ajouter les points
    for idx, row in df_sample.iterrows():
        cluster_id = row['cluster_label']
        
        # Bruit en gris
        if cluster_id == -1:
            color = 'gray'
            popup_text = "Bruit"
        else:
            color = colors[cluster_id % len(colors)]
            popup_text = f"Cluster {cluster_id}"
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=popup_text
        ).add_to(m)
    
    m.save(output_file)
    print(f"📍 Carte sauvegardée dans '{output_file}'")
    print(f"   {len(df_sample):,} points affichés sur {len(df):,} total")
    print(f"   Points gris = bruit (outliers)")

# Exemple d'usage
if __name__ == "__main__":
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')

    # 1. On arrondit pour créer une "grille" d'environ 11 mètres (4 décimales)
    df['lat_grid'] = df['lat'].round(4)
    df['long_grid'] = df['long'].round(4)

    # 2. On ne garde qu'une photo par utilisateur par case de la grille
    df_clean = df.drop_duplicates(subset=['user', 'lat_grid', 'long_grid'],keep='first')

    print(f"Nettoyage terminé : {len(df)} photos -> {len(df_clean)} photos.")
    
    # Paramètre fixe selon le notebook
    min_samples = 4
    
    # Étape 1: Trouver eps optimal avec k-distance graph
    print("=== Étape 1: Recherche de eps optimal ===")
    suggested_eps = find_optimal_eps(df, min_samples=min_samples)
    
    # Étape 2: Choisir eps à partir du graphique
    # Regardez le graphique k_distance_graph.png et choisissez la valeur au "coude"
    # Exemple: si le coude est à 0.003, utilisez cette valeur
    best_eps = suggested_eps  # Ou ajustez manuellement après avoir vu le graphique

    print(f"\n� Utilisation de eps={best_eps:.6f} avec min_samples={min_samples}")
    
    # Étape 3: Appliquer DBSCAN
    df_clustered = run_dbscan(df, eps=best_eps, min_samples=min_samples)
    
    # Étape 4: Analyser les résultats
    cluster_stats = analyze_dbscan_clusters(df_clustered)
    
    # Étape 5: Visualiser sur une carte
    visualize_dbscan_on_map(df_clustered, output_file='dbscan_lyon.html', sample_size=2000)
    
    # Optionnel: Si vous voulez tester plusieurs valeurs d'eps autour du coude
    # Décommentez les lignes suivantes:
    # print("\n=== Test de valeurs d'eps autour du coude ===")
    # eps_range = [best_eps * 0.5, best_eps * 0.75, best_eps, best_eps * 1.25, best_eps * 1.5]
    # results = test_dbscan_parameters(df, eps_range, [min_samples])
    # results.to_csv('dbscan_parameter_tests.csv', index=False)
