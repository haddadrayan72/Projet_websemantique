def load_corpus(filepath: str) -> str:
    """
    Charge le contenu brut d'un fichier texte.

    Args:
        filepath (str): chemin vers le fichier texte.

    Returns:
        str: contenu du fichier sous forme de chaîne de caractères.
    """
    # Ouverture du fichier en mode lecture avec encodage UTF-8
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


# Cette partie permet de tester le fichier en ligne de commande directement
if __name__ == "__main__":
    import sys

    # Vérifie si un chemin de fichier a été fourni en argument
    if len(sys.argv) < 2:
        print("Usage: python load_utils.py <chemin_vers_fichier.txt>")
    else:
        # Charge et affiche un extrait du texte (les 500 premiers caractères)
        text = load_corpus(sys.argv[1])
        print("Extrait du fichier chargé :")
        print(text[:500])
