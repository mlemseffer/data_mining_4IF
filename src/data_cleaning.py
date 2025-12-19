import pandas as pd
from datetime import datetime

INPUT_PATH = "../data/flickr_data2.csv"
OUTPUT_PATH = "../data/flickr_data2_cleaned.csv"
REPORT_PATH = "../data/cleaning_report.txt"

# Colonnes à retirer (si présentes): heure et minute de prise/upload
COLUMNS_TO_DROP = [
    " date_taken_minute", " date_taken_hour",
    " date_upload_minute", " date_upload_hour"
]

MIN_DATE = datetime(1900, 1, 1)
NOW = datetime.now()

# Coordonnées du rectangle de Lyon
# 45°43'11"N 4°47'36"E et 45°47'49"N 4°53'45"E
LAT_MIN = 45.719722  # 45°43'11"N
LAT_MAX = 45.796944  # 45°47'49"N
LON_MIN = 4.793333   # 4°47'36"E
LON_MAX = 4.895833   # 4°53'45"E


def main():
    print("Début du nettoyage des données...")
    
    # Lecture: tout en chaînes pour éviter les avertissements de types mixtes
    df = pd.read_csv(INPUT_PATH, low_memory=False, dtype=str)
    
    initial_count = len(df)
    print(f"Nombre initial de lignes: {initial_count}")
    
    report = []

    # 1) Suppression des doublons (par ID)
    before_duplicates = len(df)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
        removed_duplicates = before_duplicates - len(df)
        report.append(f"Doublons supprimés (id): {removed_duplicates}")
        print(f"Doublons supprimés: {removed_duplicates}")
    else:
        df = df.drop_duplicates(keep="first")
        removed_duplicates = before_duplicates - len(df)
        report.append(f"Doublons supprimés (ligne entière): {removed_duplicates}")
        print(f"Doublons supprimés: {removed_duplicates}")

    # 2) Vérification de la présence des colonnes de dates
    # Les dates sont en colonnes séparées (année, mois, jour, heure, minute)
    take_cols = [" date_taken_year", " date_taken_month", " date_taken_day", " date_taken_hour", " date_taken_minute"]
    upload_cols = [" date_upload_year", " date_upload_month", " date_upload_day", " date_upload_hour", " date_upload_minute"]
    
    required_cols = set(take_cols + upload_cols)
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"Colonnes manquantes: {missing}")

    # 3) Reconstruction des dates à partir des colonnes séparées
    print("Reconstruction des dates à partir des colonnes année/mois/jour...")
    
    # Construire les dates de prise (sans heure/minute pour simplifier)
    take_dates = pd.to_datetime(
        df[" date_taken_year"].astype(str) + "-" + 
        df[" date_taken_month"].astype(str) + "-" + 
        df[" date_taken_day"].astype(str),
        errors="coerce",
        format="%Y-%m-%d"
    )
    
    # Construire les dates d'upload (sans heure/minute pour simplifier)
    upload_dates = pd.to_datetime(
        df[" date_upload_year"].astype(str) + "-" + 
        df[" date_upload_month"].astype(str) + "-" + 
        df[" date_upload_day"].astype(str),
        errors="coerce",
        format="%Y-%m-%d"
    )

    # 4) Comptage des différents types d'éliminations
    before_date_cleaning = len(df)
    
    # Lignes avec erreurs de parsing (dates non convertibles)
    mask_parse_ok = take_dates.notna() & upload_dates.notna()
    parse_errors = (~mask_parse_ok).sum()
    
    # Parmi les dates valides, vérifier les bornes
    mask_take_valid = (take_dates >= MIN_DATE) & (take_dates <= NOW)
    mask_upload_valid = (upload_dates >= MIN_DATE) & (upload_dates <= NOW)
    
    # Vérifier que l'upload est après la prise
    mask_chronology_ok = upload_dates >= take_dates
    
    # Compter les suppressions par catégorie (sur les dates parsées correctement)
    removed_take_out = (mask_parse_ok & ~mask_take_valid).sum()
    removed_upload_out = (mask_parse_ok & mask_take_valid & ~mask_upload_valid).sum()
    removed_chronology = (mask_parse_ok & mask_take_valid & mask_upload_valid & ~mask_chronology_ok).sum()
    
    # Masque final: garder seulement les lignes cohérentes
    mask_keep = mask_parse_ok & mask_take_valid & mask_upload_valid & mask_chronology_ok
    df = df[mask_keep].reset_index(drop=True)
    
    report.append(f"Lignes supprimées (erreur parsing date): {int(parse_errors)}")
    report.append(f"Lignes supprimées (date prise hors bornes [1900-maintenant]): {int(removed_take_out)}")
    report.append(f"Lignes supprimées (date upload hors bornes [1900-maintenant]): {int(removed_upload_out)}")
    report.append(f"Lignes supprimées (upload avant prise): {int(removed_chronology)}")
    
    print(f"Lignes supprimées pour erreur parsing: {int(parse_errors)}")
    print(f"Lignes supprimées pour date prise hors bornes: {int(removed_take_out)}")
    print(f"Lignes supprimées pour date upload hors bornes: {int(removed_upload_out)}")
    print(f"Lignes supprimées pour upload avant prise: {int(removed_chronology)}")

    # 4.5) Filtre géographique - ne garder que les données dans le rectangle de Lyon
    print("Application du filtre géographique...")
    before_geo = len(df)
    
    # Convertir lat et long en numérique
    df[' lat'] = pd.to_numeric(df[' lat'], errors='coerce')
    df[' long'] = pd.to_numeric(df[' long'], errors='coerce')
    
    # Filtrer par coordonnées géographiques
    mask_geo = (
        (df[' lat'] >= LAT_MIN) & (df[' lat'] <= LAT_MAX) &
        (df[' long'] >= LON_MIN) & (df[' long'] <= LON_MAX) &
        df[' lat'].notna() & df[' long'].notna()
    )
    
    df = df[mask_geo].reset_index(drop=True)
    removed_geo = before_geo - len(df)
    
    report.append(f"Lignes supprimées (hors zone géographique Lyon): {int(removed_geo)}")
    print(f"Lignes supprimées hors zone géographique: {int(removed_geo)}")

    # 5) Suppression des colonnes inutiles
    cols_removed = [c for c in COLUMNS_TO_DROP if c in df.columns]
    
    # Supprimer aussi les colonnes Unnamed
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed:')]
    cols_removed.extend(unnamed_cols)
    
    if cols_removed:
        df = df.drop(columns=cols_removed)
        report.append(f"Colonnes supprimées: {', '.join(cols_removed)}")
        print(f"Colonnes supprimées: {', '.join(cols_removed)}")
    else:
        report.append("Colonnes supprimées: Aucune (colonnes non trouvées)")
        print("Aucune colonne à supprimer (non trouvées dans le dataset)")

    # 6) Résumé final
    final_count = len(df)
    total_removed = initial_count - final_count
    report.append(f"\nRésumé:")
    report.append(f"  Lignes initiales: {initial_count}")
    report.append(f"  Lignes finales: {final_count}")
    report.append(f"  Total supprimé: {total_removed} ({100*total_removed/initial_count:.2f}%)")

    # 7) Export
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCSV nettoyé sauvegardé: {OUTPUT_PATH}")
    print(f"Lignes finales: {final_count}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    
    print(f"Rapport sauvegardé: {REPORT_PATH}\n")
    print("="*60)
    print("RAPPORT DE NETTOYAGE")
    print("="*60)
    print("\n".join(report))


if __name__ == "__main__":
    main()
