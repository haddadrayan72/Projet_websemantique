import argparse  # Pour gérer les arguments en ligne de commande
from load_utils import load_corpus  # Chargement du texte brut depuis un fichier
from text_preprocessing import preprocess_text  # Nettoyage et tokenisation du texte
from tfidf_calculation import TFIDFProcessor  # Calcul du TF-IDF et filtrage
from vector_builder import build_document_vectors  # Création des vecteurs à partir du TF-IDF
from matrix_builder import reduce_matrix_svd  # Réduction de dimension avec SVD
from sklearn.cluster import KMeans  # Clustering KMeans
from sklearn.metrics.pairwise import cosine_similarity  # Calcul des similarités cosinus
import matplotlib.pyplot as plt  # Visualisation du graphe
import networkx as nx  # Création et manipulation de graphes

def run_pipeline(input_file, output_image, similarity_threshold=0.5):
    # 1. Chargement du texte
    raw_text = load_corpus(input_file)

    # 2. Prétraitement : nettoyage, tokenisation, etc.
    tokens = preprocess_text(raw_text)
    joined_text = [" ".join(tokens)]  # On prépare un seul document pour TF-IDF

    # 3. Calcul du TF-IDF et filtrage
    tfidf_proc = TFIDFProcessor(min_threshold=0.1)
    filtered_docs, tfidf_matrix, feature_names = tfidf_proc.compute_tfidf(joined_text)

    # 4. Création d’une liste unique des termes filtrés
    all_terms = list({term for doc in filtered_docs for term in doc})

    # 5. Construction des vecteurs document x termes (ici un seul document)
    doc_vectors = build_document_vectors(dict(enumerate(filtered_docs)), all_terms, joined_text)

    if doc_vectors.shape[1] == 0:
        print("Aucun terme pertinent après TF-IDF. Essayez un autre texte.")
        return

    # 6. Réduction de dimension avec SVD (au max 10 ou nb de colonnes)
    n_components = min(10, doc_vectors.shape[1])
    print(f"Réduction SVD à {n_components} composantes (sur {doc_vectors.shape[1]} dimensions)")
    reduced_matrix = reduce_matrix_svd(doc_vectors, n_components=n_components)

    # 7. Transposition pour appliquer le clustering sur les termes (colonnes)
    term_matrix = doc_vectors.T
    if term_matrix.shape[0] > 1:
        kmeans = KMeans(n_clusters=min(3, term_matrix.shape[0]), random_state=42)
        labels = kmeans.fit_predict(term_matrix)
    else:
        labels = [0] * term_matrix.shape[0]

    # 8. Calcul des similarités cosinus entre les termes
    term_similarity = cosine_similarity(term_matrix)

    # 9. Construction du graphe dirigé
    G = nx.DiGraph()
    for idx, term in enumerate(all_terms):
        G.add_node(term, group=labels[idx])  # chaque mot est un noeud avec une couleur (groupe)

    for i in range(len(all_terms)):
        for j in range(len(all_terms)):
            if i != j and term_similarity[i][j] >= similarity_threshold:
                G.add_edge(all_terms[i], all_terms[j], weight=round(term_similarity[i][j], 2))

    # 10. Affichage et sauvegarde du graphe
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    node_colors = [G.nodes[n]['group'] for n in G.nodes]

    plt.figure(figsize=(12, 9))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Set3,
            node_size=1500, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Graphe de similarité entre mots avec clustering")
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Graphe enregistré : {output_image}")

# Partie CLI : pour exécuter depuis le terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de génération de graphe à partir d’un texte.")
    parser.add_argument("input_file", help="Chemin vers le fichier texte")
    parser.add_argument("-o", "--output", default="output_graph.png", help="Nom de l’image en sortie")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Seuil de similarité cosinus pour créer les arêtes")
    args = parser.parse_args()
    run_pipeline(args.input_file, args.output, args.threshold)
