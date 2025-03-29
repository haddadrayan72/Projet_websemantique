def load_corpus(filepath: str) -> str:
    """
    Charge le contenu brut d'un fichier texte.

    Args:
        filepath (str): chemin vers le fichier texte.

    Returns:
        str: contenu du fichier sous forme de chaîne de caractères.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def load_multiple_corpus(filepaths: list) -> list:
    """
    Charge plusieurs fichiers texte et retourne une liste de leurs contenus.

    Args:
        filepaths (list): liste de chemins vers des fichiers texte.

    Returns:
        list: liste de chaînes de caractères, une par fichier.
    """
    return [load_corpus(path) for path in filepaths]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_utils.py <chemin1.txt> <chemin2.txt> ...")
    else:
        for path in sys.argv[1:]:
            text = load_corpus(path)
            print(f"\n--- Contenu de {path} (extrait 300 caractères) ---")
            print(text[:300])
