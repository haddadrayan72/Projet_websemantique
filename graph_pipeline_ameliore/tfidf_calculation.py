from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDFProcessor:
    def __init__(self, min_threshold=0.25):
        # Initialise le vectoriseur TF-IDF et définit un seuil pour filtrer les termes peu significatifs
        self.vectorizer = TfidfVectorizer()
        self.min_threshold = min_threshold

    def compute_tfidf(self, documents):
        """
        Calcule la matrice TF-IDF pour une liste de documents.
        Ne conserve que les termes dont le score dépasse un certain seuil.

        Args:
            documents (list[str]): liste de documents sous forme de chaînes de caractères

        Returns:
            filtered_docs (list[dict]): liste de dictionnaires {terme: tfidf} filtrés
            tfidf_matrix (np.ndarray): la matrice TF-IDF complète
            feature_names (list[str]): les noms des termes (colonnes de la matrice)
        """
        # Apprentissage du vocabulaire et transformation en matrice TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Liste des mots utilisés comme colonnes
        feature_names = self.vectorizer.get_feature_names_out()

        filtered_docs = []

        # Pour chaque document...
        for doc_idx in range(tfidf_matrix.shape[0]):
            # Conversion du vecteur sparse en tableau dense
            vector = tfidf_matrix[doc_idx].toarray().flatten()

            # Filtrage : on ne garde que les termes au-dessus du seuil
            filtered = {
                feature_names[i]: vector[i]
                for i in range(len(vector))
                if vector[i] >= self.min_threshold
            }

            filtered_docs.append(filtered)

        # Retour des résultats : dictionnaire filtré, matrice brute, noms de colonnes
        return filtered_docs, tfidf_matrix.toarray(), feature_names.tolist()


if __name__ == "__main__":
    # Exemple simple pour tester le TF-IDF
    sample_docs = [
        "snort is a network intrusion detection tool",
        "deep learning and neural network",
        "decision tree and gradient descent"
    ]

    processor = TFIDFProcessor(min_threshold=0.2)
    vectors, matrix, features = processor.compute_tfidf(sample_docs)

    print("TF-IDF filtré :", vectors)
    print("Features :", features)
