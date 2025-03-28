import numpy as np

def build_document_vectors(filtered_tfidf: dict, all_terms: list, docs: list) -> np.ndarray:
    """
    Construit une matrice de vecteurs à partir des scores TF-IDF filtrés.

    Args:
        filtered_tfidf (dict): dictionnaire {index_document: {terme: score_tfidf}}
        all_terms (list): liste globale des termes conservés (vocabulaire filtré)
        docs (list): liste des documents (utile pour conserver l’ordre)

    Returns:
        np.ndarray: matrice de vecteurs de forme (n_docs, n_termes)
    """
    vectors = []

    # Pour chaque document
    for doc_index, _ in enumerate(docs):
        doc_vector = []
        # Récupère les scores TF-IDF du document, ou dictionnaire vide si non trouvé
        doc_tfidf = filtered_tfidf.get(doc_index, {})

        # Pour chaque terme du vocabulaire, ajoute le score ou 0 si absent
        for term in all_terms:
            doc_vector.append(doc_tfidf.get(term, 0.0))

        vectors.append(doc_vector)

    return np.array(vectors)


if __name__ == "__main__":
    # Exemple simple pour tester le fonctionnement
    example_tfidf = {
        0: {'snort': 0.8, 'network': 0.4},
        1: {'svm': 0.9, 'tree': 0.3},
        2: {'math': 0.5, 'addition': 0.7}
    }
    all_terms = ['snort', 'network', 'svm', 'tree', 'math', 'addition']
    docs = ["doc0", "doc1", "doc2"]

    vectors = build_document_vectors(example_tfidf, all_terms, docs)
    print("Matrice de vecteurs :")
    print(vectors)
