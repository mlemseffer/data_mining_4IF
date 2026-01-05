import pandas as pd

# Charger le fichier nettoyé
cleaned_data_path = "../data/flickr_data2_cleaned.csv"
df = pd.read_csv(cleaned_data_path)

# Vérifier les doublons par ID
if "id" in df.columns:
    duplicate_rows = df[df.duplicated(subset=["id"], keep=False)]
    if not duplicate_rows.empty:
        print("Lignes avec des IDs en double :")
        print(duplicate_rows)
    else:
        print("Aucun ID en double trouvé.")
else:
    print("La colonne 'id' n'existe pas dans le fichier nettoyé.")