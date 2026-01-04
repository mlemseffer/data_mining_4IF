import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


def plot_dendrogram(model, lbls, title='Hierarchical Clustering Dendrogram', x_title='Samples', **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([
        model.children_,
        model.distances_,
        counts
    ]).astype(float)

    fig = plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=lbls, leaf_rotation=90)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    return fig


def hierarchical_clustering(data, labels, metric='euclidean', linkage='average', n_clusters=None, dist_thres=None):
    model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True, compute_distances=True)
    model = model.fit(data)
    txt_title = f'Hierarchical Clustering Dendrogram, linkage: {linkage}'
    f = plot_dendrogram(model=model, lbls=labels, title=txt_title, x_title='Samples')
    return model, f


def visualize_clusters_on_map(df, cluster_col, output_file='clusters_lyon_hierarchical.html', sample_size=1000):
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


def main():
    # Charger les données nettoyées
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')

    # Limiter à un échantillon aléatoire de 1000 points pour éviter les problèmes de mémoire
    sample_size = 1000
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Clustering hiérarchique sur un échantillon de {sample_size} points (sur {len(df)})")
    else:
        df_sample = df
        print(f"Clustering hiérarchique sur l'ensemble des {len(df)} points")

    X = df_sample[['lat', 'long']].values
    labels = df_sample['id'].astype(str).values

    # Tester plusieurs types de linkage et visualiser sur carte
    for link in ['complete', 'average', 'single']:
        print(f'=== Linkage: {link} ===')
        model, fig = hierarchical_clustering(X, labels, metric='euclidean', linkage=link, n_clusters=9, dist_thres=None)
        df_sample[f'cluster_{link}'] = model.labels_
        silhouette_avg = silhouette_score(X, model.labels_, metric='euclidean')
        sample_silhouette_values = silhouette_samples(X, model.labels_, metric='euclidean')
        df_sample[f'silhouette_{link}'] = sample_silhouette_values
        print(f'Linkage: {link}, silhouette score: {silhouette_avg:.3f}')

        # Visualiser les clusters sur une carte interactive
        visualize_clusters_on_map(df_sample, cluster_col=f'cluster_{link}', output_file=f'../maps/clusters_lyon_hierarchical_{link}.html', sample_size=1000)

    # Sauvegarder le DataFrame échantillon avec les clusters
    # df_sample.to_csv('../data/flickr_data2_hierarchical_sample.csv', index=False)
    # print('Résultats sauvegardés dans ../data/flickr_data2_hierarchical_sample.csv')


if __name__ == '__main__':
    main()
