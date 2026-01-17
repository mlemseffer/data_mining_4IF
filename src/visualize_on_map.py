import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial import ConvexHull

from text_mining import preprocess_dataframe, compute_tfidf_per_cluster, display_cluster_keywords


def visualize_clusters_on_map(df, output_file, sample_size=2000, show_keywords=True):
    """
    Visualise les clusters sur une carte interactive de Lyon.
    
    Args:
        df: DataFrame avec les colonnes 'lat', 'long' et 'cluster_label'
        output_file: Nom du fichier HTML de sortie
        sample_size: Nombre de points à afficher (pour la performance)
        show_keywords: Si True, affiche les mots-clés TF-IDF (clustering hybride uniquement)
    """
    print(f"\n{'='*70}")
    print("VISUALISATION DES CLUSTERS SUR CARTE")
    print(f"{'='*70}")
    
    # Calculer les mots-clés UNIQUEMENT si demandé 
    cluster_keywords = {}
    if show_keywords and 'text_merged' in df.columns:
        print("Calcul des mots-clés TF-IDF par cluster...")
        df_clustered = df[df['cluster_label'] != -1].copy()
        
        if len(df_clustered) > 0:
            cluster_keywords = compute_tfidf_per_cluster(
                df_clustered,
                cluster_col='cluster_label',
                top_n=3
            )
    
    # Créer la carte centrée sur Lyon
    center_lat = df['lat'].mean()
    center_long = df['long'].mean()
    m = folium.Map(location=[center_lat, center_long], zoom_start=12)
    
    # Générer des couleurs équidistantes dans l'espace HSV
    cluster_ids = sorted([c for c in df['cluster_label'].unique() if c != -1])
    n_clusters = len(cluster_ids)
    
    # Espacer uniformément les couleurs : prendre n_clusters couleurs équidistantes sur 360° de teinte
    colors_hex = []
    for i in range(n_clusters):
        hue = (i * 360 / n_clusters) / 360  # De 0 à 1
        # Saturation et valeur élevées pour des couleurs vives
        rgb = mcolors.hsv_to_rgb([hue, 0.85, 0.95])
        colors_hex.append(mcolors.rgb2hex(rgb))
    
    # Mapping cluster_id -> couleur hex
    color_map = {cluster_id: colors_hex[i] for i, cluster_id in enumerate(cluster_ids)}
    
    # Échantillonner si trop de points
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    print(f"Affichage de {len(df_sample):,} points sur {len(df):,}")
    
    # Ajouter les points de bruit en gris
    df_noise = df_sample[df_sample['cluster_label'] == -1]
    for row in df_noise.itertuples():
        folium.CircleMarker(
            location=[row.lat, row.long],
            radius=2,
            color='gray',
            fill=True,
            fillColor='gray',
            fillOpacity=0.3,
            popup="Bruit (outlier)"
        ).add_to(m)
    
    # Ajouter les points colorés par cluster
    df_clustered_sample = df_sample[df_sample['cluster_label'] != -1]
    for row in df_clustered_sample.itertuples():
        cluster_id = int(row.cluster_label)
        color_hex = color_map.get(cluster_id, '#808080')  # Gris par défaut si cluster inconnu
        folium.CircleMarker(
            location=[row.lat, row.long],
            radius=3,
            color=color_hex,
            fill=True,
            fillColor=color_hex,
            fillOpacity=0.7,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)
    
    # Ajouter les zones de clusters (polygones convexes)
    for cluster_id in sorted([c for c in df['cluster_label'].unique() if c != -1]):
        cluster_df = df[df['cluster_label'] == cluster_id]
        center_lat = cluster_df['lat'].mean()
        center_long = cluster_df['long'].mean()
        
        # Récupérer les 3 mots-clés les plus pertinents (si disponibles)
        keywords = cluster_keywords.get(cluster_id, [])

        popup_html = f"""<b>Cluster {cluster_id}</b><br>
            Taille: {len(cluster_df)} photos<br>
            Lat: {center_lat:.4f}<br>
            Long: {center_long:.4f}
            """
        
        if keywords:
            keywords_text = '<br>'.join([f"{i+1}. {word}" for i, (word, score) in enumerate(keywords)])
            popup_html += f"""<br>
            <br><b>Mots-clés:</b><br>
            {keywords_text}
            """
       
        # Obtenir la couleur pour ce cluster
        color_hex = color_map.get(cluster_id, '#808080')
        
        # Créer une zone englobante pour le cluster
        coords = cluster_df[['lat', 'long']].values
        
        if len(coords) >= 3:
            # Utiliser l'enveloppe convexe si assez de points
            try:
                hull = ConvexHull(coords)
                hull_points = coords[hull.vertices]
                hull_coords = [[lat, lon] for lat, lon in hull_points]
                
                # Créer un polygone avec popup et tooltip
                folium.Polygon(
                    locations=hull_coords,
                    color=color_hex,
                    fill=True,
                    fillColor=color_hex,
                    fillOpacity=0.2,
                    weight=2,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Cluster {cluster_id} ({len(cluster_df)} photos)"
                ).add_to(m)
            except Exception as e:
                # Si le convex hull échoue, utiliser un cercle
                print(f"  Convex hull échoué pour cluster {cluster_id}, utilisation d'un cercle")
                # Calculer le rayon englobant (en mètres)
                from math import radians, cos, sin, sqrt
                
                # Distance max du centre
                max_dist = 0
                for lat, lon in coords:
                    # Formule de Haversine simplifiée pour petites distances
                    dlat = radians(lat - center_lat)
                    dlon = radians(lon - center_long)
                    a = sin(dlat/2)**2 + cos(radians(center_lat)) * cos(radians(lat)) * sin(dlon/2)**2
                    c = 2 * sqrt(a)
                    dist = 6371000 * c  # Rayon de la Terre en mètres
                    max_dist = max(max_dist, dist)
                
                folium.Circle(
                    location=[center_lat, center_long],
                    radius=max_dist * 1.1,  # 10% de marge
                    color=color_hex,
                    fill=True,
                    fillColor=color_hex,
                    fillOpacity=0.2,
                    weight=2,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Cluster {cluster_id} ({len(cluster_df)} photos)"
                ).add_to(m)
        elif len(coords) == 2:
            # Pour 2 points, créer une ligne/zone entre eux
            folium.Polygon(
                locations=[[coords[0][0], coords[0][1]], [coords[1][0], coords[1][1]]],
                color=color_hex,
                weight=3,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Cluster {cluster_id} ({len(cluster_df)} photos)"
            ).add_to(m)
        else:
            # Pour 1 point, créer un cercle simple
            folium.Circle(
                location=[coords[0][0], coords[0][1]],
                radius=50,  # 50 mètres
                color=color_hex,
                fill=True,
                fillColor=color_hex,
                fillOpacity=0.3,
                weight=2,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Cluster {cluster_id} ({len(cluster_df)} photos)"
            ).add_to(m)
    
    # Sauvegarder
    m.save(output_file)
    print(f"Carte sauvegardée: {output_file}")
    print(f"{'='*70}\n")
