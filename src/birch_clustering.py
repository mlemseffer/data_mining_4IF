import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from visualize_on_map import visualize_clusters_on_map


def birch_clustering(data, n_clusters=9, threshold=0.01, branching_factor=50):
    model = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    model.fit(data)
    return model


def main():
    df = pd.read_csv('../data/flickr_data2_cleaned.csv')
    X = df[['lat', 'long']].values

    print(f"Clustering BIRCH sur {len(df)} points...")
    n_clusters = 21
    model = birch_clustering(X, n_clusters=n_clusters)
    df['birch_cluster'] = model.labels_

    silhouette_avg = silhouette_score(X, model.labels_, metric='euclidean')
    print(f"Score de silhouette moyen (BIRCH) : {silhouette_avg:.3f}")
    sample_silhouette_values = silhouette_samples(X, model.labels_, metric='euclidean')
    df['birch_silhouette'] = sample_silhouette_values

    visualize_clusters_on_map(df, output_file='../maps/clusters_birch.html', sample_size=2000, show_keywords=False, cluster_col='birch_cluster')

    df.to_csv('../data/flickr_data2_birch.csv', index=False)
    print('Résultats sauvegardés dans ../data/flickr_data2_birch.csv')


if __name__ == '__main__':
    main()
