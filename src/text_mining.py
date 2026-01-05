import re
import pandas as pd

# Stopwords français et anglais (liste manuelle)
stop_words = {
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

custom_stopwords = {'photo', 'picture', 'image', 'flickr', 'taken', 'nikon', 'canon', 'shot', 
                   'camera', 'digital', 'photos', 'images', 'pictures', 'img', 'lyon', 'france', 'europe'}

def simple_stem(word):
    """Stemming simple pour français et anglais"""
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

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
    # Tokenisation simple par split
    tokens = text.split()
    tokens = [t.strip() for t in tokens if t.strip()]
    tokens = [t for t in tokens if t not in stop_words and t not in custom_stopwords and len(t) > 2]
    # Stemming simple
    tokens = [simple_stem(t) for t in tokens]
    return tokens

def preprocess_dataframe(df, text_cols=['tags', 'title']):
    for col in text_cols:
        df[f'{col}_tokens'] = df[col].apply(preprocess_text)
    return df

# Exemple d'utilisation
if __name__ == '__main__':
    # Charger un DataFrame exemple
    df = pd.read_csv('../data/flickr_data2_hierarchical_sample.csv')
    df = preprocess_dataframe(df, text_cols=['tags', 'title'])
    print(df[['tags_tokens', 'title_tokens']].head())
