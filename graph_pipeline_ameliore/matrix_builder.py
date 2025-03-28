import numpy as np
from sklearn.decomposition import TruncatedSVD

def reduce_matrix_svd(matrix: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Réduit la dimensionnalité d’une matrice à l’aide de la SVD (Singular Value Decomposition).

    Cette fonction est utile pour simplifier les données (ex : matrice TF-IDF)
    tout en conservant l'essentiel de l'information, ce qui facilite
    la visualisation ou le clustering par exemple.

    Args:
        matrix (np.ndarray): Matrice d’entrée, typiquement documents-termes.
        n_components (int): Nombre de dimensions à conserver après réduction.

    Returns:
        np.ndarray: Nouvelle matrice avec une dimension réduite.
    """
    # Sécurité : on ne peut pas réduire à plus que le nombre de colonnes disponibles
    n_components = min(n_components, matrix.shape[1])
    
    # Initialisation et application de la SVD tronquée
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(matrix)
    
    return reduced


if __name__ == "__main__":
    # Exemple de matrice TF-IDF ou similaire
    example_matrix = np.array([
        [0.8, 0.4, 0.0],
        [0.0, 0.0, 0.9],
        [0.5, 0.3, 0.1]
    ])

    # Réduction à 2 dimensions
    reduced = reduce_matrix_svd(example_matrix, n_components=2)

    # Affichage de la matrice réduite
    print("Matrice réduite :")
    print(reduced)
