from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def find_optimal_eps(df, min_samples=15):
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
    
    print(f"\n{'='*70}")
    print("RECHERCHE DE EPS OPTIMAL (K-DISTANCE)")
    print(f"{'='*70}")
    print(f"Nombre de points analysés: {len(coords):,}")
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
    # Créer le dossier maps si besoin
    os.makedirs('../maps', exist_ok=True)
    plt.savefig('../maps/k_distance_graph.png', dpi=150)
    print("\n📊 Graphique sauvegardé : '../maps/k_distance_graph.png'")
    plt.close()
    
    # 5. Aide au diagnostic
    print(f"\n📊 Analyse des distances (en MÈTRES) :")
    percentiles = [50, 75, 85, 90, 95, 98]
    for p in percentiles:
        val_m = np.percentile(k_distances_meters, p)
        print(f"   Percentile {p:2d}%: {val_m:.2f} mètres")
    
    # 6. Suggestion (percentile 80 souvent optimal)
    suggested_eps_rad = np.percentile(k_distances_rad, 80)
    suggested_eps_m = suggested_eps_rad * 6371000
    
    print(f"\n💡 Valeur suggérée (percentile 80): {suggested_eps_m:.1f} mètres")
    print(f"   Soit eps = {suggested_eps_rad:.8f} radians")
    print(f"   Si trop de clusters -> Augmentez eps")
    print(f"   Si trop de bruit -> Augmentez min_samples")
    print(f"{'='*70}\n")
    
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

def run_dbscan(df, eps, min_samples=15, metric='haversine'):
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
    
    print(f"\n{'='*70}")
    print(f"CLUSTERING DBSCAN SPATIAL")
    print(f"{'='*70}")
    
    print(f"\n--- Paramètres DBSCAN utilisés ---")
    print(f"eps = {eps:.8f} (radians, soit {eps * 6371000:.1f} mètres)")
    print(f"min_samples = {min_samples}")
    print(f"metric = {metric}")
    
    if metric == 'haversine':
        # Convertir en radians pour haversine
        coords_rad = np.radians(coords)
        eps_m = eps * 6371000  # Conversion en mètres
        print(f"Paramètres:")
        print(f"  - Eps: {eps:.8f} radians ({eps_m:.1f} mètres)")
        print(f"  - Min samples: {min_samples}")
        print(f"  - Métrique: Haversine (distances GPS réelles)")
        
        print(f"\nClustering DBSCAN...")
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
        df['cluster_label'] = dbscan.fit_predict(coords_rad)
    else:
        print(f"Paramètres:")
        print(f"  - Eps: {eps:.6f}")
        print(f"  - Min samples: {min_samples}")
        print(f"  - Métrique: {metric}")
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        df['cluster_label'] = dbscan.fit_predict(coords)
    
    # Statistiques
    n_clusters = len(set(df['cluster_label'])) - (1 if -1 in df['cluster_label'].values else 0)
    n_noise = (df['cluster_label'] == -1).sum()
    
    print(f"\nRésultats:")
    print(f"  - Clusters trouvés: {n_clusters}")
    print(f"  - Points de bruit: {n_noise:,} ({n_noise/len(df)*100:.1f}%)")
    print(f"  - Points dans clusters: {len(df) - n_noise:,} ({(len(df)-n_noise)/len(df)*100:.1f}%)")
    
    # Silhouette score
    if n_clusters > 1:
        mask_clustered = df['cluster_label'] != -1
        if mask_clustered.sum() > 0:
            try:
                silhouette = silhouette_score(coords_rad if metric == 'haversine' else coords, 
                                             df.loc[mask_clustered, 'cluster_label'],
                                             metric=metric)
                print(f"  - Score de silhouette: {silhouette:.3f}")
            except:
                pass
    
    # 8. Statistiques par cluster
    print(f"\n{'='*70}")
    print("STATISTIQUES PAR CLUSTER")
    print(f"{'='*70}")
    print(f"{'Cluster':<10} {'Taille':<10} {'Lat moy':<12} {'Long moy':<12} {'% du total':<12}")
    print('-'*70)
    
    if n_noise > 0:
        percentage = (n_noise / len(df)) * 100
        print(f"{'BRUIT (-1)':<10} {n_noise:<10} {'-':<12} {'-':<12} {percentage:<12.1f}%")
    
    for cluster_id in sorted([c for c in set(df['cluster_label']) if c != -1]):
        cluster_df = df[df['cluster_label'] == cluster_id]
        size = len(cluster_df)
        lat_mean = cluster_df['lat'].mean()
        long_mean = cluster_df['long'].mean()
        percentage = (size / len(df)) * 100
        print(f"{cluster_id:<10} {size:<10} {lat_mean:<12.4f} {long_mean:<12.4f} {percentage:<12.1f}%")
    
    print('-'*70)
    
    cluster_sizes = df[df['cluster_label'] != -1]['cluster_label'].value_counts()
    if len(cluster_sizes) > 0:
        print(f"Taille min (hors bruit): {cluster_sizes.min()}")
        print(f"Taille max (hors bruit): {cluster_sizes.max()}")
        print(f"Taille moyenne: {cluster_sizes.mean():.1f}")
        print(f"Écart-type: {cluster_sizes.std():.1f}")
    
    print(f"{'='*70}\n")
    
    # Avertissement si résultats suspects
    if n_clusters > 100:
        print(f"⚠️  ATTENTION: Nombre de clusters très élevé ({n_clusters})!")
        print(f"   → Augmentez eps pour fusionner les clusters")
    elif n_clusters < 3:
        print(f"⚠️  ATTENTION: Très peu de clusters ({n_clusters})")
        print(f"   → Diminuez eps pour détecter plus de zones\n")
    
    return df

def analyze_dbscan_clusters(df):
    """
    Analyse détaillée des résultats du clustering DBSCAN.
    
    Args:
        df: DataFrame avec la colonne 'cluster_label'
    """
    print(f"\n{'='*70}")
    print("ANALYSE DÉTAILLÉE DES CLUSTERS")
    print(f"{'='*70}")
    
    # Séparer le bruit des clusters
    noise = df[df['cluster_label'] == -1]
    clustered = df[df['cluster_label'] != -1]
    
    print(f"\nRépartition globale:")
    print(f"  - Points de bruit: {len(noise):,} ({len(noise)/len(df)*100:.1f}%)")
    print(f"  - Points clusterisés: {len(clustered):,} ({len(clustered)/len(df)*100:.1f}%)")
    
    if len(clustered) > 0:
        # Nombre de photos par cluster
        cluster_counts = clustered['cluster_label'].value_counts().sort_index()
        print(f"  - Nombre de clusters: {len(cluster_counts)}")
        
        # Top 10 clusters par taille
        print(f"\nTop 10 clusters par taille:")
        top_clusters = cluster_counts.head(10)
        for cluster_id, count in top_clusters.items():
            cluster_df = df[df['cluster_label'] == cluster_id]
            lat_mean = cluster_df['lat'].mean()
            long_mean = cluster_df['long'].mean()
            print(f"  Cluster {cluster_id:3d}: {count:5,} photos - Centre: ({lat_mean:.4f}, {long_mean:.4f})")
        
        print(f"{'='*70}\n")
        
        return clustered
    else:
        print("\n⚠️  Aucun cluster trouvé!")
        print(f"{'='*70}\n")
        return None

def visualize_dbscan_on_map(df, output_file='../maps/dbscan_lyon.html', sample_size=2000):
    """
    Visualise les clusters DBSCAN sur une carte interactive.
    Le bruit est affiché en gris.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_label'
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher
    """
    import folium
    
    print(f"\n{'='*70}")
    print("VISUALISATION DES CLUSTERS SUR CARTE")
    print(f"{'='*70}")
    
    # Créer une carte centrée sur Lyon
    center_lat = df['lat'].mean()
    center_long = df['long'].mean()
    m = folium.Map(location=[center_lat, center_long], zoom_start=12)
    
    # Couleurs pour les clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'brown', 'black', 'white']
    
    # Échantillonner
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    print(f"Affichage de {len(df_sample):,} points sur {len(df):,}")
    
    # Ajouter les points de bruit en gris
    df_noise = df_sample[df_sample['cluster_label'] == -1]
    for idx, row in df_noise.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=2,
            color='gray',
            fill=True,
            fillColor='gray',
            fillOpacity=0.3,
            popup="Bruit (outlier)"
        ).add_to(m)
    
    # Ajouter les points colorés par cluster
    df_clustered = df_sample[df_sample['cluster_label'] != -1]
    for idx, row in df_clustered.iterrows():
        cluster_id = int(row['cluster_label'])
        color = colors[cluster_id % len(colors)]
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)
    
    # Ajouter les centres des clusters avec marqueurs
    for cluster_id in sorted([c for c in df['cluster_label'].unique() if c != -1]):
        cluster_df = df[df['cluster_label'] == cluster_id]
        center_lat = cluster_df['lat'].mean()
        center_long = cluster_df['long'].mean()
        
        popup_html = f"""<b>Cluster {cluster_id}</b><br>
        Taille: {len(cluster_df):,} photos<br>
        Lat: {center_lat:.4f}<br>
        Long: {center_long:.4f}
        """
        
        folium.Marker(
            location=[center_lat, center_long],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=colors[cluster_id % len(colors)], icon='info-sign')
        ).add_to(m)
    
    m.save(output_file)
    print(f"Carte sauvegardée: {output_file}")
    print(f"{'='*70}\n")

# Exemple d'usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "CLUSTERING DBSCAN - LYON")
    print("="*70 + "\n")
    
    # Chargement des données
    print("Chargement des données...")
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    print(f"Données chargées: {len(df):,} photos")
    
    # Échantillonnage pour performance
    sample_size = 5000
    if len(df) > sample_size:
        print(f"Échantillonnage de {sample_size:,} photos pour le clustering...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Paramètres optimisés
    min_samples = 15  # Plus robuste que 4
    
    # Étape 1: Trouver eps optimal avec k-distance graph
    suggested_eps = find_optimal_eps(df, min_samples=min_samples)
    
    # Étape 2: Utiliser eps suggéré (ou ajuster manuellement si nécessaire)
    best_eps = suggested_eps
    
    # Étape 3: Appliquer DBSCAN
    df_clustered = run_dbscan(df, eps=best_eps, min_samples=min_samples)
    
    # Étape 4: Analyser les résultats
    clustered_data = analyze_dbscan_clusters(df_clustered)
    
    # Étape 5: Visualiser sur une carte
    visualize_dbscan_on_map(df_clustered, output_file='../maps/dbscan_lyon.html', sample_size=2000)
    
    # Étape 6: Sauvegarder les résultats
    output_file = '../data/flickr_data2_dbscan_clustering.csv'
    df_clustered.to_csv(output_file, index=False)
    print(f"Résultats sauvegardés: {output_file}\n")
    
    # Optionnel: Test de différents paramètres
    # Décommentez pour tester plusieurs valeurs d'eps:
    # print("\n" + "="*70)
    # print("TEST DE PARAMÈTRES")
    # print("="*70)
    # eps_range = [best_eps * 0.5, best_eps * 0.75, best_eps, best_eps * 1.25, best_eps * 1.5]
    # results = test_dbscan_parameters(df, eps_range, [min_samples])
    # results.to_csv('../data/dbscan_parameter_tests.csv', index=False)
    # print(f"\nRésultats des tests sauvegardés: '../data/dbscan_parameter_tests.csv'\n")
