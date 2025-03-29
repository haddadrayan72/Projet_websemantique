from rdflib import Graph, Namespace, Literal, RDF

def extract_rdf_triples(tokens, window_size=2):
    """
    Construit un graphe RDF à partir d'une liste de tokens avec cooccurrence.

    Args:
        tokens (list): Liste de mots après prétraitement
        window_size (int): Fenêtre de cooccurrence

    Returns:
        rdflib.Graph: Graphe RDF contenant les triples
    """
    g = Graph()
    EX = Namespace("http://example.org/word/")
    g.bind("ex", EX)

    for i, source in enumerate(tokens):
        source_uri = EX[source]
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            target = tokens[j]
            target_uri = EX[target]
            g.add((source_uri, EX.cooccursWith, target_uri))

    return g
