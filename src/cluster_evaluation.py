"""
Évaluation de la qualité des clusters avec score de silhouette
Fonctions réutilisables pour comparer différentes méthodes de clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score, silhouette_samples

from color_utils import get_matplotlib_color_for_cluster


def evaluate_clustering(data, labels, metric='euclidean', method_name='Clustering'):
    """
    Évalue un clustering avec le coefficient de silhouette.
    
    Args:
        data: Features normalisées (array ou DataFrame)
        labels: Labels des clusters (-1 pour le bruit)
        metric: Métrique de distance ('euclidean', 'haversine', etc.)
        method_name: Nom de la méthode de clustering (pour affichage)
        
    Returns:
        dict avec:
            - silhouette_avg: Score moyen de silhouette
            - silhouette_samples: Scores individuels par échantillon
            - n_clusters: Nombre de clusters (hors bruit)
            - n_noise: Nombre de points de bruit
            - cluster_scores: Scores moyens par cluster
    """
    # Filtrer le bruit si présent
    mask = labels != -1
    data_clean = data[mask] if isinstance(data, np.ndarray) else data.iloc[mask]
    labels_clean = labels[mask]
    
    n_clusters = len(set(labels_clean))
    n_noise = np.sum(~mask)
    
    if n_clusters > 1 and len(labels_clean) > 0:
        silhouette_avg = silhouette_score(data_clean, labels_clean, metric=metric)
        sample_silhouette_values = np.full(len(labels), np.nan)
        sample_silhouette_values[mask] = silhouette_samples(data_clean, labels_clean, metric=metric)
        
        cluster_scores = {}
        for cluster_id in set(labels_clean):
            cluster_mask = labels == cluster_id
            cluster_scores[cluster_id] = sample_silhouette_values[cluster_mask].mean()
    else:
        silhouette_avg = np.nan
        sample_silhouette_values = np.full(len(labels), np.nan)
        cluster_scores = {}
    
    print(f"\n{'='*70}")
    print(f"ÉVALUATION: {method_name}")
    print(f"{'='*70}")
    print(f"  - Nombre de clusters: {n_clusters}")
    print(f"  - Points de bruit: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"  - Score de silhouette moyen: {silhouette_avg:.3f}")
    print(f"\n=== Qualité du clustering ===")
    print("  > 0.7  : Clustering excellent")
    print("  0.5-0.7: Clustering bon")
    print("  0.25-0.5: Clustering moyen")
    print("  < 0.25 : Clustering faible")
    
    return {
        'silhouette_avg': silhouette_avg,
        'silhouette_samples': sample_silhouette_values,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_scores': cluster_scores
    }


def plot_silhouette(sample_silhouette_values, silhouette_avg, labels, show_plot = True,
                   file_name = "silhouette.png", n_clusters=None,  
                   title='Silhouette Plot', figsize=(10, 6)):
    """
    Crée un graphique de silhouette pour visualiser la qualité du clustering.
    
    Args:
        sample_silhouette_values: Scores de silhouette individuels
        silhouette_avg: Score moyen de silhouette
        labels: Labels des clusters
        show_plot: Si True, affiche le graphique
        file_name: Nom du fichier pour sauvegarder le graphique
        n_clusters: Nombre de clusters (calculé automatiquement si None)
        title: Titre du graphique
        figsize: Taille de la figure
        
    Returns:
        fig: Figure matplotlib
    """
    # Filtrer les valeurs NaN (bruit)
    mask = ~np.isnan(sample_silhouette_values)
    sample_silhouette_clean = sample_silhouette_values[mask]
    labels_clean = labels[mask]
    
    if n_clusters is None:
        n_clusters = len(set(labels_clean))
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    cluster_ids_sorted = sorted(set(labels_clean))
    cluster_to_index = {cid: idx for idx, cid in enumerate(cluster_ids_sorted)}
    
    y_lower = 10
    for i in cluster_ids_sorted:
        ith_cluster_values = sample_silhouette_clean[labels_clean == i]
        ith_cluster_values = np.sort(ith_cluster_values)
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Couleur pour ce cluster (utilise le même système que la carte)
        cluster_index = cluster_to_index[i]
        color = get_matplotlib_color_for_cluster(cluster_index, n_clusters)
        
        # Remplir la zone de silhouette
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        # Étiquette pour le cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
        
        y_lower = y_upper + 10
    
    # Ligne verticale pour le score moyen
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
              label=f'Moyenne: {silhouette_avg:.3f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Coefficient de Silhouette', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_yticks([])
    ax.legend(loc='best')
    
    plt.tight_layout()

    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/' + file_name, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Graphique de silhouette sauvegarde dans '../plots/{file_name}'")

    if show_plot:
        plt.show()

    return fig


def compare_clustering_methods(results_dict, figsize=(12, 6)):
    """
    Compare visuellement plusieurs méthodes de clustering.
    
    Args:
        results_dict: Dictionnaire {nom_méthode: résultats de evaluate_clustering()}
        figsize: Taille de la figure
        
    Returns:
        fig: Figure matplotlib avec graphiques de comparaison
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Graphique 1: Scores de silhouette moyens
    methods = list(results_dict.keys())
    scores = [results_dict[m]['silhouette_avg'] for m in methods]
    n_clusters_list = [results_dict[m]['n_clusters'] for m in methods]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    ax1 = axes[0]
    bars = ax1.bar(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Score de Silhouette', fontsize=12)
    ax1.set_title('Comparaison des Scores de Silhouette', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Seuil acceptable (0.5)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Ajouter le nombre de clusters sur les barres
    for i, (bar, n_clust) in enumerate(zip(bars, n_clusters_list)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{n_clust} clusters\n{scores[i]:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # Graphique 2: Nombre de clusters et bruit
    ax2 = axes[1]
    x = np.arange(len(methods))
    width = 0.35
    
    clusters = [results_dict[m]['n_clusters'] for m in methods]
    noise = [results_dict[m]['n_noise'] for m in methods]
    
    ax2.bar(x - width/2, clusters, width, label='Clusters', color='skyblue', edgecolor='black')
    ax2.bar(x + width/2, noise, width, label='Bruit', color='lightcoral', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Nombre de points', fontsize=12)
    ax2.set_title('Clusters vs Bruit', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_comparison_table(results_dict):
    """
    Crée un tableau de comparaison des méthodes de clustering.
    
    Args:
        results_dict: Dictionnaire {nom_méthode: résultats de evaluate_clustering()}
        
    Returns:
        DataFrame avec les résultats comparatifs
    """
    data = []
    for method, results in results_dict.items():
        data.append({
            'Méthode': method,
            'Score Silhouette': f"{results['silhouette_avg']:.3f}",
            'Nb Clusters': results['n_clusters'],
            'Nb Bruit': results['n_noise'],
            '% Bruit': f"{results['n_noise']/(results['n_clusters']*50 + results['n_noise'])*100:.1f}%"  # Approximation
        })
    
    df = pd.DataFrame(data)
    
    print(f"\n{'='*70}")
    print("TABLEAU COMPARATIF DES MÉTHODES DE CLUSTERING")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}\n")
    
    return df


def print_cluster_silhouette_scores(results_dict):
    """
    Affiche les scores de silhouette par cluster pour chaque méthode.
    
    Args:
        results_dict: Dictionnaire {nom_méthode: résultats de evaluate_clustering()}
    """
    for method, results in results_dict.items():
        print(f"\n{'='*70}")
        print(f"SCORES PAR CLUSTER - {method}")
        print(f"{'='*70}")
        
        cluster_scores = results['cluster_scores']
        if cluster_scores:
            print(f"{'Cluster':<15} {'Score Silhouette':<20}")
            print('-'*35)
            for cluster_id, score in sorted(cluster_scores.items()):
                print(f"{cluster_id:<15} {score:<20.3f}")
        else:
            print("Aucun score disponible")
        print('-'*70)


def evaluate_and_compare(data, clustering_methods, metric='euclidean', 
                        show_plots=True, show_silhouette_plots=False):
    """
    Fonction complète pour évaluer et comparer plusieurs méthodes de clustering.
    
    Args:
        data: Features normalisées
        clustering_methods: Dictionnaire {nom_méthode: labels}
        metric: Métrique de distance
        show_plots: Si True, affiche les graphiques de comparaison
        show_silhouette_plots: Si True, affiche les graphiques de silhouette individuels
        
    Returns:
        results_dict: Résultats détaillés de toutes les évaluations
        comparison_df: DataFrame avec tableau comparatif
    """
    results_dict = {}
    
    for method_name, labels in clustering_methods.items():
        results = evaluate_clustering(data, labels, metric=metric, method_name=method_name)
        results_dict[method_name] = results
        
        if show_silhouette_plots and not np.isnan(results['silhouette_avg']):
            fig = plot_silhouette(
                results['silhouette_samples'],
                results['silhouette_avg'],
                labels,
                results['n_clusters'],
                title=f"Silhouette Plot - {method_name}"
            )
            plt.show()
    
    comparison_df = create_comparison_table(results_dict)
    
    print_cluster_silhouette_scores(results_dict)
    
    if show_plots:
        fig = compare_clustering_methods(results_dict)
        plt.show()
    
    return results_dict, comparison_df


if __name__ == '__main__':
    print("Module d'évaluation de clustering chargé.")
    print("Utilisez evaluate_and_compare() pour comparer plusieurs méthodes.")
