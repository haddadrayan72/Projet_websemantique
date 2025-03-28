import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Téléchargement automatique des ressources nécessaires à la première exécution
nltk.download('punkt')       # Tokeniseur de phrases/mots
nltk.download('stopwords')   # Liste de mots vides (ex: the, is, etc.)
nltk.download('wordnet')     # Base pour la lemmatisation

def preprocess_text(text: str) -> list:
    """
    Fonction de nettoyage et de préparation du texte :
    - Supprime les balises HTML
    - Convertit en minuscules
    - Supprime la ponctuation
    - Tokenise le texte (sépare les mots)
    - Supprime les stopwords
    - Lemmatisation (réduction à la racine)

    Args:
        text (str): texte brut

    Returns:
        list: liste de tokens nettoyés et normalisés
    """
    # Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # Passage en minuscules
    text = text.lower()

    # Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenisation
    tokens = word_tokenize(text)

    # Initialisation des stopwords et lemmatiseur
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Suppression des stopwords et lemmatisation des tokens
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word.isalpha()  # Supprime aussi les chiffres
    ]

    return tokens


if __name__ == "__main__":
    # Exemple de test
    sample = "<p>This is <b>Snort</b>, a network intrusion-detection tool that detects attacks!</p>"
    print("Texte d'origine :", sample)
    print("Tokens nettoyés :", preprocess_text(sample))
