import argparse
from load_utils import load_corpus  # Chargement du fichier texte brut
from text_preprocessing import preprocess_text  # Nettoyage, tokenisation, lemmatisation
from tfidf_calculation import TFIDFProcessor  # Calcul TF-IDF avec filtrage
from vector_builder import build_document_vectors  # Construction des vecteurs document x terme
from matrix_builder import reduce_matrix_svd  # Réduction dimensionnelle via SVD
from rdf_builder import extract_rdf_triples  # Extraction des triplets RDF
from rdflib import Graph
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # Pour le clustering Louvain (RDF)

# ------------------ RDF : Visualisation ------------------

def visualize_rdf_graph(rdf_graph: Graph, output_image: str):
    """
    Transforme un graphe RDF en visualisation PNG avec NetworkX.
    """
    G = nx.DiGraph()

    # Création du graphe orienté depuis les triplets RDF
    for s, p, o in rdf_graph:
        G.add_edge(s.split('/')[-1], o.split('/')[-1], label=p.split('/')[-1])

    # Clustering Louvain pour des couleurs de groupes
    partition = community_louvain.best_partition(G.to_undirected())
    node_colors = [partition[n] for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, font_size=6, node_size=1000, arrows=True, node_color=node_colors, cmap=plt.cm.Set3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Graphe RDF (cooccurrences)")
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"[✔] Graphe RDF enregistré : {output_image}")


# ------------------ RDF : Pipeline ------------------

def run_rdf_pipeline(input_file, output_image, window_size=2):
    """
    Extrait un graphe RDF à partir d’un texte, le convertit en PNG et sauvegarde aussi le fichier .ttl.
    """
    raw_text = load_corpus(input_file)
    tokens = preprocess_text(raw_text)

    # Extraction des triplets RDF à partir d'une fenêtre glissante
    rdf_graph = extract_rdf_triples(tokens, window_size=window_size)

    # Sauvegarde graphique
    visualize_rdf_graph(rdf_graph, output_image)

    # Sauvegarde du fichier RDF au format Turtle
    ttl_file = output_image.replace(".png", ".ttl")
    rdf_graph.serialize(destination=ttl_file, format="turtle")
    print(f"[✔] RDF exporté au format Turtle : {ttl_file}")


# ------------------ SIMILARITÉ : Pipeline ------------------

def run_similarity_pipeline(input_files, output_image, similarity_threshold=0.5, top_k=5):
    """
    Génère un graphe de similarité sémantique à partir de plusieurs fichiers texte.
    """
    corpus = []
    for file in input_files:
        raw_text = load_corpus(file)
        tokens = preprocess_text(raw_text)
        corpus.append(" ".join(tokens))

    # TF-IDF avec filtrage (min_threshold)
    tfidf_proc = TFIDFProcessor(min_threshold=0.1)
    filtered_docs, _, _ = tfidf_proc.compute_tfidf(corpus)

    # Vocabulaire unique filtré
    all_terms = list({term for doc in filtered_docs for term in doc})

    # Création de la matrice vecteurs
    doc_vectors = build_document_vectors(dict(enumerate(filtered_docs)), all_terms, corpus)

    # Sécurité si pas assez de termes
    if doc_vectors.shape[1] == 0:
        print("Aucun terme pertinent après TF-IDF. Essayez un autre texte.")
        return

    # Réduction dimensionnelle
    n_components = min(10, doc_vectors.shape[1])
    print(f"[INFO] Réduction SVD à {n_components} composantes (sur {doc_vectors.shape[1]} dimensions)")
    reduced_matrix = reduce_matrix_svd(doc_vectors, n_components=n_components)

    # Transposition : clustering sur les mots (termes)
    term_matrix = doc_vectors.T

    # Clustering KMeans
    if term_matrix.shape[0] > 1:
        kmeans = KMeans(n_clusters=min(3, term_matrix.shape[0]), random_state=42)
        labels = kmeans.fit_predict(term_matrix)
    else:
        labels = [0] * term_matrix.shape[0]

    # Calcul des similarités entre mots
    term_similarity = cosine_similarity(term_matrix)

    # Construction du graphe
    G = nx.DiGraph()
    for idx, term in enumerate(all_terms):
        G.add_node(term, group=labels[idx])

    for i in range(len(all_terms)):
        similarities = list(enumerate(term_similarity[i]))
        similarities = sorted(
            [(j, score) for j, score in similarities if i != j and score >= similarity_threshold],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        for j, score in similarities:
            G.add_edge(all_terms[i], all_terms[j], weight=round(score, 2))

    # Affichage
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    node_colors = [G.nodes[n]['group'] for n in G.nodes]
    edge_widths = [d['weight'] * 3 for _, _, d in G.edges(data=True)]

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Set3,
            node_size=1500, font_weight='bold', arrows=True, width=edge_widths)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Graphe de similarité entre mots (top-k connexions + pondération)")
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"[✔] Graphe de similarité enregistré : {output_image}")


# ------------------ Interface CLI ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un graphe RDF (1 texte) ou un graphe de similarité (2+ textes).")
    parser.add_argument("input_files", nargs='+', help="Un ou plusieurs fichiers texte")
    parser.add_argument("-o", "--output", default="output_graph.png", help="Nom de l’image en sortie")
    parser.add_argument("-w", "--window", type=int, default=2, help="Fenêtre de cooccurrence pour RDF")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Seuil de similarité pour le graphe")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top connexions par mot")
    args = parser.parse_args()

    if len(args.input_files) == 1:
        run_rdf_pipeline(args.input_files[0], args.output, args.window)
    else:
        run_similarity_pipeline(args.input_files, args.output, args.threshold, args.top_k)
