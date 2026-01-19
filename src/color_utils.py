"""
Utilitaires pour la gestion des couleurs des clusters
Assure la cohérence des couleurs entre la visualisation sur carte et les graphiques
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def get_color_palette(n_clusters):
    """
    Génère une palette de couleurs distinctes pour n clusters.
    
    Args:
        n_clusters: Nombre de clusters
        
    Returns:
        dict: Mapping {cluster_id: color_hex} avec cluster_id triés
    """
    if n_clusters <= 0:
        return {}
    
    # Utiliser différentes palettes selon le nombre de clusters
    if n_clusters <= 10:
        # Palette tab10 (10 couleurs distinctes)
        colors = [mcolors.rgb2hex(plt.cm.tab10(i)) for i in range(n_clusters)]
    elif n_clusters <= 20:
        # Palette tab20 (20 couleurs distinctes)
        colors = [mcolors.rgb2hex(plt.cm.tab20(i)) for i in range(n_clusters)]
    else:
        # Pour plus de 20 clusters : générer des couleurs avec bonne distribution
        # Utiliser plusieurs teintes avec variation de luminosité/saturation
        colors = []
        for i in range(n_clusters):
            # Espacer les teintes sur 360° en évitant le jaune-vert (60-120°)
            # Diviser en 2 plages : 0-60° (rouge-orange) et 120-360° (vert-violet)
            hue_fraction = i / n_clusters
            
            if hue_fraction < 0.15:
                # Rouge à orange (0-60°)
                hue = hue_fraction * 400  # 0 à 60
            else:
                # Cyan à rouge (120-360°)
                hue = 120 + (hue_fraction - 0.15) * 282  # 120 à 360
            
            # Alterner saturation et luminosité pour plus de contraste
            saturation = 0.85 if i % 2 == 0 else 0.65
            value = 0.95 if (i // 2) % 2 == 0 else 0.75
            
            rgb = mcolors.hsv_to_rgb([hue / 360, saturation, value])
            colors.append(mcolors.rgb2hex(rgb))
    
    return colors


def get_color_for_cluster(cluster_id, cluster_ids_sorted, colors_palette):
    """
    Récupère la couleur hex pour un cluster donné.
    
    Args:
        cluster_id: ID du cluster
        cluster_ids_sorted: Liste triée de tous les IDs de clusters
        colors_palette: Liste des couleurs générées par get_color_palette()
        
    Returns:
        str: Couleur au format hex (ex: '#FF5733')
    """
    try:
        index = cluster_ids_sorted.index(cluster_id)
        return colors_palette[index]
    except (ValueError, IndexError):
        # Couleur par défaut si cluster inconnu
        return '#808080'  # Gris


def get_matplotlib_color_for_cluster(cluster_id, n_clusters):
    """
    Récupère une couleur matplotlib rgba pour le graphique de silhouette.
    Utilise la même logique que get_color_palette pour cohérence.
    
    Args:
        cluster_id: ID du cluster (position dans l'ordre trié)
        n_clusters: Nombre total de clusters
        
    Returns:
        tuple: Couleur RGBA pour matplotlib
    """
    colors_palette = get_color_palette(n_clusters)
    
    if cluster_id < len(colors_palette):
        hex_color = colors_palette[cluster_id]
        # Convertir hex en RGB
        rgb = mcolors.hex2color(hex_color)
        return rgb + (1.0,)  # Ajouter alpha=1.0
    else:
        return (0.5, 0.5, 0.5, 1.0)  # Gris par défaut


def create_cluster_color_map(cluster_ids_sorted):
    """
    Crée un mapping complet cluster_id -> couleur pour tous les clusters.
    
    Args:
        cluster_ids_sorted: Liste triée des IDs de clusters (sans -1)
        
    Returns:
        dict: {cluster_id: color_hex}
    """
    n_clusters = len(cluster_ids_sorted)
    colors_palette = get_color_palette(n_clusters)
    
    color_map = {}
    for i, cluster_id in enumerate(cluster_ids_sorted):
        color_map[cluster_id] = colors_palette[i]
    
    return color_map


if __name__ == '__main__':
    for n in [5, 15, 30, 60]:
        print(f"\n{n} clusters:")
        colors = get_color_palette(n)
        print(f"  Première couleur: {colors[0]}")
        print(f"  Dernière couleur: {colors[-1]}")
        print(f"  Total: {len(colors)} couleurs")
