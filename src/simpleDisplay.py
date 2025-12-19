import pandas as pd
import folium
from folium.plugins import MarkerCluster

def generate_lyon_map(csv_path):
    # 1. Chargement des données
    df = pd.read_csv(csv_path)
    
    print(f"Nombre total de points avant nettoyage: {len(df)}")
    # 2. Nettoyage de base 
    # Suppression des doublons basés sur l'ID de la photo
    df = df.drop_duplicates(subset=['id'])
    print(f"Nombre de points après suppression des doublons: {len(df)}")

    # 3. Création de la carte centrée sur Lyon 
    lyon_map = folium.Map(location=[45.75, 4.85], zoom_start=13)
    
    # Utilisation d'un MarkerCluster pour éviter de faire ramer le navigateur
    # avec 400 000 points 
    marker_cluster = MarkerCluster().add_to(lyon_map)

    # Supprimer les lignes avec lat/long manquantes
    df = df.dropna(subset=[' lat', ' long'])

    # 4. Ajout des points
    for idx, row in df.iterrows():
         folium.Marker(
             location=[row[' lat'], row[' long']],
             popup=f"Photo ID: {row['id']}",
         ).add_to(marker_cluster)

    # 5. Sauvegarde 
    lyon_map.save('lyon_tourism_map.html')
    print(f"Carte générée avec {len(df)} points nettoyés. Ouvrez 'lyon_tourism_map.html' dans votre navigateur.")

if __name__ == "__main__":
    generate_lyon_map('../data/flickr_data2.csv')