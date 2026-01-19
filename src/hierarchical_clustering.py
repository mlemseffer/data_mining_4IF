import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

from visualize_on_map import visualize_clusters_on_map

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


def run_hierarchical_clustering(df, n_clusters, linkage='complete', show_dendrogram=False):
    """
    Effectue un clustering hiérarchique sur les données.
    
    Args:
        df: DataFrame avec colonnes 'lat', 'long'
        n_clusters: nombre de clusters souhaités
        linkage: méthode de linkage ('complete', 'average', 'single')
        show_dendrogram: afficher le dendrogramme ou non
        
    Returns:
        df: DataFrame avec colonne 'cluster' ajoutée
        model: modèle AgglomerativeClustering entraîné
        silhouette: score de silhouette
    """
    print(f"\n{'='*70}")
    print(f"CLUSTERING HIÉRARCHIQUE - {linkage.upper()}")
    print(f"{'='*70}")
    print(f"Nombre de clusters: {n_clusters}")
    print(f"Linkage: {linkage}")
    
    X = df[['lat', 'long']].values
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage=linkage,
        compute_full_tree=show_dendrogram,
        compute_distances=show_dendrogram
    )
    
    df['cluster'] = model.fit_predict(X)
    
    silhouette_avg = silhouette_score(X, df['cluster'])
    print(f"Score de silhouette: {silhouette_avg:.3f}")
    
    if show_dendrogram:
        labels = df['id'].astype(str).values if 'id' in df.columns else None
        txt_title = f'Hierarchical Clustering Dendrogram, linkage: {linkage}'
        plot_dendrogram(model=model, lbls=labels, title=txt_title, x_title='Samples')
    
    return df, model, silhouette_avg


def hierarchical_clustering(data, labels, metric='euclidean', linkage='average', n_clusters=None, dist_thres=None):
    model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True, compute_distances=True)
    model = model.fit(data)
    txt_title = f'Hierarchical Clustering Dendrogram, linkage: {linkage}'
    f = plot_dendrogram(model=model, lbls=labels, title=txt_title, x_title='Samples')
    return model, f

def main():
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')

    sample_size = 2000
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Clustering hiérarchique sur un échantillon de {sample_size} points (sur {len(df)})")
    else:
        df_sample = df
        print(f"Clustering hiérarchique sur l'ensemble des {len(df)} points")

    X = df_sample[['lat', 'long']].values
    labels = df_sample['id'].astype(str).values

    for link in ['complete', 'average', 'single']:
        print(f'=== Linkage: {link} ===')
        model, fig = hierarchical_clustering(X, labels, metric='euclidean', linkage=link, n_clusters=20, dist_thres=None)
        df_sample[f'cluster_{link}'] = model.labels_
        silhouette_avg = silhouette_score(X, model.labels_, metric='euclidean')
        sample_silhouette_values = silhouette_samples(X, model.labels_, metric='euclidean')
        df_sample[f'silhouette_{link}'] = sample_silhouette_values
        print(f'Linkage: {link}, silhouette score: {silhouette_avg:.3f}')

        visualize_clusters_on_map(
            df_sample, output_file=f'../maps/clusters_hierarchical_{link}.html',
            sample_size=1000,
            show_keywords=False,
            cluster_col=f'cluster_{link}'
        )

    df_sample.to_csv('../data/flickr_data2_hierarchical_sample.csv', index=False)
    print('Résultats sauvegardés dans ../data/flickr_data2_hierarchical_sample.csv')


if __name__ == '__main__':
    main()
