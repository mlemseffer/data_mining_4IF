import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt


def birch_clustering(data, n_clusters=9, threshold=0.01, branching_factor=50):
    model = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    model.fit(data)
    return model


def main():
    # Charger les données nettoyées
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    X = df[['lat', 'long']].values

    print(f"Clustering BIRCH sur {len(df)} points...")
    n_clusters = 12  # À adapter selon vos besoins
    model = birch_clustering(X, n_clusters=n_clusters)
    df['birch_cluster'] = model.labels_

    # Évaluation
    silhouette_avg = silhouette_score(X, model.labels_, metric='euclidean')
    print(f"Score de silhouette moyen (BIRCH) : {silhouette_avg:.3f}")
    sample_silhouette_values = silhouette_samples(X, model.labels_, metric='euclidean')
    df['birch_silhouette'] = sample_silhouette_values

    # Visualisation sur carte interactive (folium)
    visualize_clusters_on_map(df, cluster_col='birch_cluster', output_file='../maps/clusters_lyon_birch.html', sample_size=2000)

    # Sauvegarder les résultats
    # df.to_csv('../data/flickr_data2_birch.csv', index=False)
    # print('Résultats sauvegardés dans ../data/flickr_data2_birch.csv')


def visualize_clusters_on_map(df, cluster_col, output_file='../maps/clusters_lyon_birch.html', sample_size=1000):
    """
    Visualise les clusters sur une carte interactive de Lyon.
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et cluster_col
        cluster_col: nom de la colonne des labels de cluster
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher (pour la performance)
    """
    import folium
    print(f"\n=== Visualisation des clusters ({cluster_col}) ===")
    m = folium.Map(location=[45.75, 4.85], zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'gray', 'black', 'lightgray', 'white', 'brown']
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    for idx, row in df_sample.iterrows():
        cluster_id = row[cluster_col]
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=colors[int(cluster_id) % len(colors)],
            fill=True,
            fillColor=colors[int(cluster_id) % len(colors)],
            fillOpacity=0.6,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)
    m.save(output_file)
    print(f"Carte sauvegardée dans '{output_file}'")
    print(f"{len(df_sample):,} points affichés sur {len(df):,} total")


if __name__ == '__main__':
    main()
