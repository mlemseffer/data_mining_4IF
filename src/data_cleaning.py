import pandas as pd
from datetime import datetime
import re

INPUT_PATH = "../data/flickr_data2.csv"
OUTPUT_PATH = "../data/flickr_data2_cleaned.csv"
REPORT_PATH = "../data/cleaning_report.txt"

# Colonnes à retirer (si présentes): heure et minute de prise/upload
COLUMNS_TO_DROP = [
    "date_taken_minute", "date_taken_hour",
    "date_upload_minute", "date_upload_hour"
]

MIN_DATE = datetime(1900, 1, 1)
NOW = datetime.now()

# Coordonnées du rectangle de Lyon
# 45°43'11"N 4°47'36"E et 45°47'49"N 4°53'45"E
LAT_MIN = 45.719722  # 45°43'11"N
LAT_MAX = 45.796944  # 45°47'49"N
LON_MIN = 4.793333   # 4°47'36"E
LON_MAX = 4.895833   # 4°53'45"E

# Stopwords français et anglais pour nettoyage textuel
STOP_WORDS = {
    # Anglais
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't",
    # Français
    'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 
    'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 
    'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 
    'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 
    'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'suis', 
    'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 
    'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 
    'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 
    'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'eu', 'eue', 'eues', 'eus', 
    'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 
    'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 
    'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 
    'eût', 'eussions', 'eussiez', 'eussent', 'ceci', 'cela', 'celà', 'cet', 'cette', 'ici', 'ils', 
    'les', 'leurs', 'quel', 'quels', 'quelle', 'quelles', 'sans', 'soi'
}

# Stopwords personnalisés (mots trop génériques)
CUSTOM_STOPWORDS = {
    'camera', 'canon', 'des', 'digital', 'europe', 'flickr', 'flickriosapp', 
    'flickrmobile', 'francia', 'france', 'image', 'images', 'img', 
    'instagram', 'lyon', 'nikon', 'photo', 'photos', 'picture', 
    'pictures', 'shot', 'taken', 'uploaded'
}


def simple_stem(word):
    """
    Stemming simple pour français et anglais.
    Réduit les mots à leur racine en supprimant les suffixes courants.
    """
    # Suffixes anglais courants
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('ly'):
        return word[:-2]
    if word.endswith('ness'):
        return word[:-4]
    if word.endswith('ment'):
        return word[:-4]
    # Suffixes français courants
    if word.endswith('tion'):
        return word[:-4]
    if word.endswith('sion'):
        return word[:-4]
    if word.endswith('able'):
        return word[:-4]
    if word.endswith('ible'):
        return word[:-4]
    if word.endswith('ique'):
        return word[:-4]
    if word.endswith('eur'):
        return word[:-3]
    if word.endswith('euse'):
        return word[:-4]
    if word.endswith('ait'):
        return word[:-3]
    if word.endswith('aient'):
        return word[:-6]
    if word.endswith('er'):
        return word[:-2]
    if word.endswith('é'):
        return word[:-1]
    if word.endswith('és'):
        return word[:-2]
    if word.endswith('ées'):
        return word[:-3]
    return word


def normalize_text(text):
    """
    Nettoie et normalise un champ texte.
    - Convertit en minuscules
    - Supprime caractères spéciaux (garde lettres et accents)
    - Supprime les stopwords
    - Applique le stemming
    
    Returns:
        str: Texte nettoyé (mots séparés par espaces)
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Minuscules
    text = text.lower()
    
    # Supprimer caractères spéciaux (garder lettres avec accents)
    text = re.sub(r'[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    
    # Tokenisation simple
    tokens = text.split()
    tokens = [t.strip() for t in tokens if t.strip()]
    
    # Filtrer stopwords et mots courts
    tokens = [t for t in tokens 
              if t not in STOP_WORDS 
              and t not in CUSTOM_STOPWORDS 
              and len(t) > 2]
    
    # Stemming
    tokens = [simple_stem(t) for t in tokens]
    
    # Retourner comme string
    return ' '.join(tokens)


def clean_text_fields(df):
    """
    Nettoie les champs textuels (tags, title, description) et crée text_merged.
    
    Args:
        df: DataFrame avec colonnes de texte
        
    Returns:
        df: DataFrame avec text_merged ajoutée
    """
    print("\n5. Nettoyage textuel...")
    
    # Normaliser les colonnes de texte si elles existent
    text_columns = ['tags', 'title', 'description']
    existing_cols = [col for col in text_columns if col in df.columns]
    
    if not existing_cols:
        print("   ⚠ Aucune colonne textuelle trouvée")
        df['text_merged'] = ""
        return df
    
    # Créer text_merged directement sans créer de colonnes intermédiaires
    text_parts = []
    
    for col in existing_cols:
        # Normaliser le texte
        normalized = df[col].fillna('').apply(normalize_text)
        text_parts.append(normalized)
    
    # Concaténer avec espace
    if text_parts:
        df['text_merged'] = text_parts[0]
        for part in text_parts[1:]:
            df['text_merged'] = df['text_merged'] + ' ' + part
        
        # Nettoyer espaces multiples
        df['text_merged'] = df['text_merged'].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
        df['text_merged'] = ""
    
    # Statistiques finales
    non_empty_merged = (df['text_merged'] != "").sum()
    pct_merged = (non_empty_merged / len(df) * 100) if len(df) > 0 else 0
    
    # Calculer le nombre moyen de mots
    mask_with_text = df['text_merged'] != ""
    word_counts = df.loc[mask_with_text, 'text_merged'].str.split().str.len()
    avg_words = word_counts.mean() if len(word_counts) > 0 else 0
    
    print(f"   → {non_empty_merged:,}/{len(df):,} photos avec texte ({pct_merged:.1f}%)")
    print(f"   → Moyenne: {avg_words:.1f} mots/photo")
    
    return df


def main():
    print("\n" + "="*70)
    print(" "*20 + "DATA CLEANING - FLICKR LYON")
    print("="*70 + "\n")
    
    # Lecture: tout en chaînes pour éviter les avertissements de types mixtes
    df = pd.read_csv(INPUT_PATH, low_memory=False, dtype=str)
    
    # Supprimer les espaces au début et à la fin des noms de colonnes
    df.columns = df.columns.str.strip()
    
    initial_count = len(df)
    print(f"Dataset initial: {initial_count:,} lignes")
    
    report = []

    # 1) Suppression des doublons (par lignes identiques puis par photo_id)
    print("\n1. Suppression des doublons...")
    before_duplicates = len(df)
    
    # Supprimer espaces dans toutes les colonnes
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Doublons de lignes complètes
    df = df.drop_duplicates(keep="first")
    
    # Doublons par photo_id (garder première occurrence)
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep="first")
    
    removed_duplicates = before_duplicates - len(df)
    report.append(f"Duplicates removed: {removed_duplicates}")
    print(f"   → {removed_duplicates:,} doublons supprimés")

    # 2) Vérification de la présence des colonnes de dates
    print("\n2. Validation et nettoyage des dates...")
    # Les dates sont en colonnes séparées (année, mois, jour, heure, minute)
    take_cols = ["date_taken_year", "date_taken_month", "date_taken_day", "date_taken_hour", "date_taken_minute"]
    upload_cols = ["date_upload_year", "date_upload_month", "date_upload_day", "date_upload_hour", "date_upload_minute"]
    
    required_cols = set(take_cols + upload_cols)
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"Missing columns: {missing}")

    # 3) Reconstruction des dates à partir des colonnes séparées
    # Construire les dates de prise (sans heure/minute pour simplifier)
    take_dates = pd.to_datetime(
        df["date_taken_year"].astype(str) + "-" + 
        df["date_taken_month"].astype(str) + "-" + 
        df["date_taken_day"].astype(str),
        errors="coerce",
        format="%Y-%m-%d"
    )
    
    # Construire les dates d'upload (sans heure/minute pour simplifier)
    upload_dates = pd.to_datetime(
        df["date_upload_year"].astype(str) + "-" + 
        df["date_upload_month"].astype(str) + "-" + 
        df["date_upload_day"].astype(str),
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
    
    total_date_removed = before_date_cleaning - len(df)
    report.append(f"Rows removed (date issues): {int(total_date_removed)}")
    print(f"   → {total_date_removed:,} lignes avec problèmes de dates supprimées")

    # 4.5) Filtre géographique - ne garder que les données dans le rectangle de Lyon
    print("\n3. Filtrage géographique...")
    before_geo = len(df)
    
    # Convertir lat et long en numérique
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    
    # Filtrer par coordonnées géographiques
    mask_geo = (
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['long'] >= LON_MIN) & (df['long'] <= LON_MAX) &
        df['lat'].notna() & df['long'].notna()
    )
    
    df = df[mask_geo].reset_index(drop=True)
    removed_geo = before_geo - len(df)
    
    report.append(f"Rows removed (outside Lyon area): {int(removed_geo)}")
    print(f"   → {removed_geo:,} lignes hors zone Lyon supprimées")
    
    # Doublons exacts basés sur photo_id + lat + lon + date
    print("\n4. Suppression des doublons exacts (id + coords + dates)...")
    before_dup = len(df)
    
    # Reconstruire les dates pour la déduplication
    df['date_taken'] = pd.to_datetime(
        df["date_taken_year"].astype(str) + "-" + 
        df["date_taken_month"].astype(str) + "-" + 
        df["date_taken_day"].astype(str),
        errors="coerce"
    )
    
    df = df.drop_duplicates(subset=['id', 'lat', 'long', 'date_taken'], keep='first')
    removed_exact_dup = before_dup - len(df)
    report.append(f"Exact duplicates removed: {int(removed_exact_dup)}")
    print(f"   → {removed_exact_dup:,} doublons exacts supprimés")

    # 5) Nettoyage textuel
    df = clean_text_fields(df)
    
    # 6) Suppression des colonnes inutiles
    print("\n6. Suppression des colonnes inutiles...")
    cols_removed = [c for c in COLUMNS_TO_DROP if c in df.columns]
    
    # Supprimer aussi les colonnes Unnamed et les colonnes texte intermédiaires
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed:')]
    cols_removed.extend(unnamed_cols)
    
    # Supprimer tags, title (originaux) car on a text_merged
    text_cols_to_remove = ['tags', 'title', 'description', 'date_taken']
    for col in text_cols_to_remove:
        if col in df.columns:
            cols_removed.append(col)
    
    if cols_removed:
        df = df.drop(columns=cols_removed)
        print(f"   → {len(cols_removed)} colonnes supprimées")
    
    # 7) Validation post-nettoyage
    print("\n7. Validation du dataset...")
    validation_errors = []
    
    # Check 1: Pas de NaN dans colonnes critiques
    critical_cols = ['id', 'user', 'lat', 'long']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                validation_errors.append(f"{col}: {nan_count} NaN détectés")
    
    # Check 2: GPS valides
    if 'lat' in df.columns and 'long' in df.columns:
        invalid_gps = (
            (df['lat'] < -90) | (df['lat'] > 90) |
            (df['long'] < -180) | (df['long'] > 180)
        ).sum()
        if invalid_gps > 0:
            validation_errors.append(f"GPS invalides: {invalid_gps}")
    
    # Check 3: Pas de doublons photo_id
    if 'id' in df.columns:
        dup_count = df['id'].duplicated().sum()
        if dup_count > 0:
            validation_errors.append(f"Doublons photo_id: {dup_count}")
    
    if validation_errors:
        print("   ⚠ Erreurs de validation détectées:")
        for error in validation_errors:
            print(f"      - {error}")
    else:
        print("   ✓ Validation réussie")
    
    # Ajouter statistiques texte au rapport
    non_empty_text = (df['text_merged'] != "").sum()
    pct_text = (non_empty_text / len(df) * 100) if len(df) > 0 else 0
    report.append(f"Rows with text: {non_empty_text} ({pct_text:.1f}%)")

    # 8) Résumé final    # 8) Résumé final
    final_count = len(df)
    total_removed = initial_count - final_count
    report.append(f"Initial rows: {initial_count}")
    report.append(f"Final rows: {final_count}")
    report.append(f"Total removed: {total_removed} ({100*total_removed/initial_count:.2f}%)")

    # 9) Export
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n{'='*70}")
    print("RÉSUMÉ")
    print(f"{'='*70}")
    print(f"Lignes initiales : {initial_count:,}")
    print(f"Lignes finales   : {final_count:,}")
    print(f"Lignes supprimées: {total_removed:,} ({100*total_removed/initial_count:.1f}%)")
    print(f"\nFichier sauvegardé: {OUTPUT_PATH}")
    print(f"{'='*70}\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()