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

from visualize_on_map import visualize_clusters_on_map

from cluster_evaluation import evaluate_clustering, plot_silhouette

def hdbscan_clustering(df, min_cluster_size=50, min_samples=10, spatial_weight=0.7, text_weight=0.3):
    """
    Clustering HDBSCAN combinant position géographique et contenu textuel.
    
    Args:
        df: DataFrame avec lat, long, text_merged
        min_cluster_size: taille minimale d'un cluster
        min_samples: nombre minimum d'échantillons dans un voisinage
        spatial_weight: poids des features spatiales (0-1)
        text_weight: poids des features textuelles (0-1)
        
    Returns:
        df: DataFrame avec colonne 'cluster_label' ajoutée
        clusterer: modèle HDBSCAN entraîné
        vectorizer: TfidfVectorizer utilisé
        feature_info: dict avec informations sur les features
    """
    print(f"\n{'='*70}")
    print(f"CLUSTERING HDBSCAN (Spatial + Textuel)")
    print(f"{'='*70}")
    print(f"Paramètres:")
    print(f"  - Taille min cluster: {min_cluster_size}")
    print(f"  - Min samples: {min_samples}")
    print(f"  - Poids spatial: {spatial_weight:.1%}")
    print(f"  - Poids textuel: {text_weight:.1%}")
    print(f"  - Nombre de photos: {len(df)}")
    print(f"  - Métrique: Euclidienne (après normalisation)")
    
    # 1. Prétraiter le texte si text_merged pas déjà tokenisé
    if 'text_merged_tokens' not in df.columns:
        print("\nPrétraitement du texte...")
        df = preprocess_dataframe(df)
    
    # 2. Features spatiales normalisées avec scaling fort pour Lyon
    print("\nExtraction des features spatiales...")
    spatial_features = df[['lat', 'long']].values
    
    # Normalisation Z-score avec scaling plus agressif pour renforcer la géographie
    scaler = StandardScaler()
    spatial_scaled = scaler.fit_transform(spatial_features)
    
    # Multiplier par un facteur pour accentuer les différences spatiales
    # Cela aide HDBSCAN à mieux séparer les zones géographiques
    spatial_scaled = spatial_scaled * 3.0
    
    print(f"  - Features spatiales: {spatial_scaled.shape}")
    print(f"  - Scaling factor: 3.0 (renforce les distances géographiques)")
    
    # 3. Features textuelles (TF-IDF) depuis text_merged
    print("\nExtraction des features textuelles (TF-IDF)...")
    
    # Vérifier que text_merged existe
    if 'text_merged' not in df.columns:
        print("  ⚠ Colonne text_merged absente, utilisation de texte vide")
        df['text_merged'] = ""
    
    # Filtrer les textes vides
    valid_texts = df['text_merged'].apply(lambda x: len(str(x).strip()) > 0)
    print(f"  - Textes valides: {valid_texts.sum()} / {len(df)}")
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limiter à 100 mots les plus importants
        min_df=5,          # Mot doit apparaître dans au moins 5 documents
        max_df=0.7         # Mot ne doit pas apparaître dans plus de 70% des documents
    )
    
    text_features = vectorizer.fit_transform(df['text_merged'].fillna('')).toarray()
    print(f"  - Features textuelles: {text_features.shape}")
    print(f"  - Vocabulaire: {len(vectorizer.get_feature_names_out())} mots")
    
    # 4. Combiner les features avec pondération
    print("\nCombinaison des features...")
    combined_features = np.hstack([
        spatial_scaled * spatial_weight,
        text_features * text_weight
    ])
    print(f"  - Features combinées: {combined_features.shape}")
    
    # 5. Clustering HDBSCAN avec paramètres optimisés
    print(f"\nClustering HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass - sélection plus robuste
        cluster_selection_epsilon=0.0,   # Pas de seuil de distance fixe
        alpha=1.0,                        # Contrôle la "robustesse" aux outliers
        allow_single_cluster=False        # Force plusieurs clusters
    )
    
    cluster_labels = clusterer.fit_predict(combined_features)
    df['cluster_label'] = cluster_labels
    
    # 6. Statistiques de clustering
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nRésultats:")
    print(f"  - Clusters trouvés: {n_clusters}")
    print(f"  - Points de bruit: {n_noise} ({n_noise/len(df)*100:.1f}%)")
    print(f"  - Points dans clusters: {len(df) - n_noise} ({(len(df)-n_noise)/len(df)*100:.1f}%)")
    
    # Silhouette score (seulement sur points non-bruit)
    if n_clusters > 1:
        mask_clustered = cluster_labels != -1
        if mask_clustered.sum() > 0:
            silhouette_avg = silhouette_score(
                combined_features[mask_clustered], 
                cluster_labels[mask_clustered]
            )
            print(f"  - Score de silhouette: {silhouette_avg:.3f}")
    
    # 7. Statistiques par cluster
    print(f"\n{'='*70}")
    print("STATISTIQUES PAR CLUSTER")
    print(f"{'='*70}")
    print(f"{'Cluster':<10} {'Taille':<10} {'Lat moy':<12} {'Long moy':<12} {'% du total':<12}")
    print('-'*70)
    
    # Afficher le bruit en premier
    if n_noise > 0:
        percentage = (n_noise / len(df)) * 100
        print(f"{'BRUIT (-1)':<10} {n_noise:<10} {'-':<12} {'-':<12} {percentage:<12.1f}%")
    
    # Afficher les clusters
    for cluster_id in sorted([c for c in set(cluster_labels) if c != -1]):
        cluster_df = df[df['cluster_label'] == cluster_id]
        size = len(cluster_df)
        lat_mean = cluster_df['lat'].mean()
        long_mean = cluster_df['long'].mean()
        percentage = (size / len(df)) * 100
        print(f"{cluster_id:<10} {size:<10} {lat_mean:<12.4f} {long_mean:<12.4f} {percentage:<12.1f}%")
    
    print('-'*70)
    
    # Stats sur les clusters (hors bruit)
    cluster_sizes = df[df['cluster_label'] != -1]['cluster_label'].value_counts()
    if len(cluster_sizes) > 0:
        print(f"Taille min (hors bruit): {cluster_sizes.min()}")
        print(f"Taille max (hors bruit): {cluster_sizes.max()}")
        print(f"Taille moyenne: {cluster_sizes.mean():.1f}")
        print(f"Écart-type: {cluster_sizes.std():.1f}")
    
    # Informations sur les features
    feature_info = {
        'scaler': scaler,
        'vectorizer': vectorizer,
        'spatial_weight': spatial_weight,
        'text_weight': text_weight,
        'vocabulary': vectorizer.get_feature_names_out().tolist(),
        'n_spatial_features': spatial_scaled.shape[1],
        'n_text_features': text_features.shape[1],
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    return df, clusterer, vectorizer, feature_info

def analyze_cluster_content(df):
    """
    Analyse le contenu textuel de chaque cluster avec TF-IDF.
    
    Args:
        df: DataFrame avec 'cluster_label' et text_merged
    """
    print(f"\n{'='*70}")
    print("ANALYSE DU CONTENU TEXTUEL PAR CLUSTER (TF-IDF)")
    print(f"{'='*70}")
    
    # Analyser seulement les vrais clusters (pas le bruit)
    df_clustered = df[df['cluster_label'] != -1].copy()
    
    if len(df_clustered) == 0:
        print("Aucun cluster trouvé (tous les points sont du bruit)")
        return
    
    cluster_keywords = compute_tfidf_per_cluster(
        df_clustered,
        cluster_col='cluster_label',
        top_n=10
    )
    
    display_cluster_keywords(
        cluster_keywords,
        title="Top 10 mots-clés par cluster (TF-IDF) - HDBSCAN"
    )


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
        allow_single_cluster=False
    )
    
    cluster_labels = clusterer.fit_predict(coords_rad)
    df['cluster_spatial_hdbscan'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nRésultats:")
    print(f"  - Clusters trouvés: {n_clusters}")
    print(f"  - Points de bruit: {n_noise} ({n_noise/len(df)*100:.1f}%)")
    
    return df


def compare_hdbscan_vs_hybrid(df, min_cluster_size=50, min_samples=15):
    """
    Compare HDBSCAN spatial vs hybride avec visualisation.
    
    Args:
        df: DataFrame avec données et clusters déjà calculés
        min_cluster_size: taille min pour clustering spatial
        min_samples: nombre min d'échantillons
    """
    print(f"\n{'='*70}")
    print("COMPARAISON: HDBSCAN SPATIAL vs HYBRIDE")
    print(f"{'='*70}")
    
    # Clustering spatial HDBSCAN
    df = hdbscan_spatial_only(df, min_cluster_size=min_cluster_size, min_samples=min_samples)
    
    # Visualisation du clustering spatial (SANS mots-clés)
    print("\nGénération de la carte pour clustering spatial...")
    # Créer une copie temporaire pour la visualisation
    df_temp = df.copy()
    visualize_clusters_on_map(
        df_temp,
        output_file='../maps/clusters_hdbscan_spatial.html',
        show_keywords=True,
        cluster_col='cluster_spatial_hdbscan'
    )

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
        file_name='hdbscan_spatial_silhouette.png',
        show_plot=True
    )
    
    # Statistiques
    if 'cluster_label' in df.columns and 'cluster_spatial_hdbscan' in df.columns:
        print(f"\n{'='*70}")
        print("RÉSULTATS DE COMPARAISON")
        print(f"{'='*70}")
        
        # Compter clusters et bruit
        hybrid_clusters = len(set(df['cluster_label'])) - (1 if -1 in df['cluster_label'].values else 0)
        hybrid_noise = (df['cluster_label'] == -1).sum()
        
        spatial_clusters = len(set(df['cluster_spatial_hdbscan'])) - (1 if -1 in df['cluster_spatial_hdbscan'].values else 0)
        spatial_noise = (df['cluster_spatial_hdbscan'] == -1).sum()
        
        print(f"{'Méthode':<20} {'Clusters':<15} {'Bruit':<15} {'% Bruit':<15}")
        print('-'*70)
        print(f"{'Spatial':<20} {spatial_clusters:<15} {spatial_noise:<15} {spatial_noise/len(df)*100:<15.1f}")
        print(f"{'Hybride':<20} {hybrid_clusters:<15} {hybrid_noise:<15} {hybrid_noise/len(df)*100:<15.1f}")
        
        # Tailles de clusters
        print(f"\n{'='*70}")
        print("ÉQUILIBRE DES TAILLES (hors bruit)")
        print(f"{'='*70}")
        print(f"{'Méthode':<20} {'Min':<10} {'Max':<10} {'Moy':<10} {'Écart-type':<12}")
        print('-'*70)
        
        spatial_sizes = df[df['cluster_spatial_hdbscan'] != -1]['cluster_spatial_hdbscan'].value_counts()
        if len(spatial_sizes) > 0:
            print(f"{'Spatial':<20} {spatial_sizes.min():<10} {spatial_sizes.max():<10} "
                  f"{spatial_sizes.mean():<10.1f} {spatial_sizes.std():<12.1f}")
        
        hybrid_sizes = df[df['cluster_label'] != -1]['cluster_label'].value_counts()
        if len(hybrid_sizes) > 0:
            print(f"{'Hybride':<20} {hybrid_sizes.min():<10} {hybrid_sizes.max():<10} "
                  f"{hybrid_sizes.mean():<10.1f} {hybrid_sizes.std():<12.1f}")
        
        print(f"{'='*70}\n")


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
    SPATIAL_WEIGHT = 0.7    # Poids spatial augmenté
    TEXT_WEIGHT = 0.3       # Poids textuel réduit
    
    # 1. Clustering HDBSCAN hybride
    df, clusterer, vectorizer, feature_info = hdbscan_clustering(
        df,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        spatial_weight=SPATIAL_WEIGHT,
        text_weight=TEXT_WEIGHT
    )
    
    # 2. Visualisation sur carte (avec mots-clés car hybride)
    visualize_clusters_on_map(df, output_file='../maps/clusters_hdbscan_hybrid.html', show_keywords=True)
    
    # 3. Analyse du contenu textuel
    analyze_cluster_content(df)
    
    # 4. Comparaison avec clustering spatial pur
    compare_hdbscan_vs_hybrid(df, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
    
    # 5. Sauvegarder les résultats
    output_file = '../data/flickr_data2_hdbscan_clustering.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Résultats sauvegardés: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
