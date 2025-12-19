import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import haversine_distances

EARTH_KM = 6371.0


def load_coords(path, lat_col="lat", lon_col="long", nrows=None):
    # read without forcing dtypes so we can detect column names
    df = pd.read_csv(path, nrows=nrows)
    cols = list(df.columns)

    def norm_name(s):
        return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())

    mapping = {norm_name(c): c for c in cols}

    # candidate normalized names
    lat_cands = ["lat", "latitude", "y", "gpslat"]
    lon_cands = ["lon", "long", "longitude", "lng", "x", "gpslon"]

    # try provided names first
    selected_lat = None
    selected_lon = None
    if lat_col in df.columns:
        selected_lat = lat_col
    elif norm_name(lat_col) in mapping:
        selected_lat = mapping[norm_name(lat_col)]
    else:
        for c in lat_cands:
            if c in mapping:
                selected_lat = mapping[c]
                break

    if lon_col in df.columns:
        selected_lon = lon_col
    elif norm_name(lon_col) in mapping:
        selected_lon = mapping[norm_name(lon_col)]
    else:
        for c in lon_cands:
            if c in mapping:
                selected_lon = mapping[c]
                break

    if selected_lat is None or selected_lon is None:
        raise KeyError(
            f"Colonnes lat/lon introuvables. Colonnes disponibles: {cols}. "
            "Précisez --lat-col et --lon-col si besoin."
        )

    # coerce to numeric, drop NA and rename for rest of the script
    df[selected_lat] = pd.to_numeric(df[selected_lat], errors="coerce")
    df[selected_lon] = pd.to_numeric(df[selected_lon], errors="coerce")
    df = df.dropna(subset=[selected_lat, selected_lon]).reset_index(drop=True)
    df = df.rename(columns={selected_lat: "lat", selected_lon: "long"})
    coords = df[["lat", "long"]].to_numpy(dtype=float)
    return df, coords


def haversine_distance_matrix(coords_deg):
    # coords_deg: (n,2) array [lat, lon] in degrees
    coords_rad = np.radians(coords_deg)
    # haversine_distances returns radians; multiply by EARTH_KM to get km
    D = haversine_distances(coords_rad) * EARTH_KM
    return D


def run_agglomerative(coords_deg, n_clusters=None, distance_km=None, linkage="average"):
    n = len(coords_deg)
    if n == 0:
        raise ValueError("Aucun point à clusteriser.")
    # compute pairwise haversine distance matrix (km)
    D = haversine_distance_matrix(coords_deg)
    if distance_km is not None:
        # AgglomerativeClustering supports distance_threshold (requires n_clusters=None)
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage=linkage,
            distance_threshold=float(distance_km),
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=int(n_clusters) if n_clusters is not None else 2,
            affinity="precomputed",
            linkage=linkage,
        )
    labels = model.fit_predict(D)
    return labels


def reduce_with_kmeans(coords_deg, n_centroids, random_state=0):
    """Réduit coords (deg) en n_centroids centroïdes via KMeans sur représentation 3D sphérique."""
    lat = np.radians(coords_deg[:, 0])
    lon = np.radians(coords_deg[:, 1])
    # 3D cartesian on sphere (radius 1)
    X = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    kmeans = KMeans(n_clusters=int(n_centroids), random_state=random_state).fit(X)
    centers = kmeans.cluster_centers_
    # normaliser puis convertir en lat/lon degrés
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers_unit = centers / norms
    cx, cy, cz = centers_unit[:, 0], centers_unit[:, 1], centers_unit[:, 2]
    cent_lat = np.degrees(np.arcsin(cz))
    cent_lon = np.degrees(np.arctan2(cy, cx))
    return np.column_stack((cent_lat, cent_lon)), kmeans


def assign_points_to_centroids(coords_deg, centroids_deg):
    """Retourne pour chaque point l'indice du centroïde le plus proche (haversine)."""
    # Utilise matrice distances haversine
    D = haversine_distance_matrix(np.vstack((coords_deg, centroids_deg)))
    n = len(coords_deg)
    m = len(centroids_deg)
    # D has shape (n+m, n+m); extract distances between points (0:n) and centroids (n:n+m)
    D_pts_cent = D[:n, n:n+m]
    return np.argmin(D_pts_cent, axis=1)


def save_results(df, labels, output_csv):
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out.to_csv(output_csv, index=False)
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Hierarchical clustering on lat/long (haversine)")
    parser.add_argument("--input", "-i", default="../../data/flickr_data2_cleaned.csv")
    parser.add_argument("--output", "-o", default="../data/flickr_data2_clustered.csv")
    parser.add_argument("--n-clusters", type=int, default=None, help="Nombre de clusters (mutuellement exclusif avec --distance-km)")
    parser.add_argument("--distance-km", type=float, default=None, help="Seuil de coupure en km (n_clusters=None)")
    parser.add_argument("--sample-size", type=int, default=None, help="Si fourni, échantillonne ce nombre de points (utile pour grands jeux de données)")
    parser.add_argument("--map", action="store_true", help="Génère une carte HTML avec les clusters (requires folium)")
    parser.add_argument("--lat-col", type=str, default="lat", help="Nom de la colonne latitude dans le CSV")
    parser.add_argument("--lon-col", type=str, default="long", help="Nom de la colonne longitude dans le CSV")
    parser.add_argument("--approx-centroids", type=int, default=None, help="Réduire d'abord via KMeans à ce nb de centroïdes, puis hierarchique (utile pour gros jeux).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée non trouvé: {input_path}")

    df, coords = load_coords(input_path, lat_col=args.lat_col, lon_col=args.lon_col, nrows=None)
    n = len(df)
    if args.sample_size and args.sample_size < n:
        df = df.sample(args.sample_size, random_state=0).reset_index(drop=True)
        coords = df[["lat", "long"]].to_numpy(dtype=float)
        n = len(df)

    # Si gros jeu et option de réduction demandée, faire la réduction
    if n > 5000 and args.approx_centroids:
        print(f"Réduction à {args.approx_centroids} centroïdes via KMeans...")
        centroids_deg, kmeans = reduce_with_kmeans(coords, args.approx_centroids, random_state=0)
        # Faire l'agglomératif sur centroïdes
        labels_cent = run_agglomerative(centroids_deg, n_clusters=args.n_clusters, distance_km=args.distance_km)
        # Affecter chaque point au centroïde le plus proche puis mapper aux labels_cent
        idx_cent = assign_points_to_centroids(coords, centroids_deg)
        labels = labels_cent[idx_cent]
    else:
        if n > 5000:
            raise RuntimeError(f"Trop de points ({n}). Échantillonnez (--sample-size) ou utilisez --approx-centroids N ou un autre algorithme (DBSCAN/KMeans).")
        if args.n_clusters is None and args.distance_km is None:
            raise ValueError("Spécifier --n-clusters ou --distance-km")
        labels = run_agglomerative(coords, n_clusters=args.n_clusters, distance_km=args.distance_km)
    out_df = save_results(df, labels, args.output)
    print(f"Sauvegardé: {args.output} (n={len(out_df)})")

    if args.map:
        try:
            import folium
        except Exception as e:
            print("folium non installé. Faites: pip install folium")
            return
        mean_lat = out_df["lat"].mean()
        mean_lon = out_df["long"].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        # palette simple
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        for _, row in out_df.iterrows():
            c = palette[int(row["cluster"]) % len(palette)]
            folium.CircleMarker(
                location=(row["lat"], row["long"]),
                radius=3,
                color=c,
                fill=True,
                fill_opacity=0.7,
            ).add_to(m)
        map_path = Path(args.output).with_suffix(".html")
        m.save(str(map_path))
        print(f"Carte sauvegardée: {map_path}")


if __name__ == "__main__":
    main()