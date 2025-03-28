# Projet Web-Sémantique  
## Graphe de Similarité entre Mots

Ce projet génère un **graphe dirigé** entre les mots d’un texte, basé sur leur similarité sémantique.  
Il s’appuie sur du **prétraitement linguistique**, le **TF-IDF**, la **réduction SVD**, le **clustering KMeans**, et une **visualisation graphique avec NetworkX**.

---

## 🧠 Fonctionnalités

- Nettoyage, tokenisation et lemmatisation du texte
- Calcul des scores TF-IDF avec filtrage par seuil
- Construction de vecteurs de mots
- Réduction de dimension via SVD (TruncatedSVD)
- Clustering des mots avec KMeans
- Génération d’un graphe de similarité pondéré

---

## 📁 Arborescence du projet


```bash
graph_pipeline_ameliore/ │ 
├── main_ameliore.py # Script principal CLI 
├── load_utils.py # Chargement du texte 
├── text_preprocessing.py # Nettoyage et tokenisation 
├── tfidf_calculation.py # Calcul TF-IDF filtré 
├── vector_builder.py # Construction des vecteurs 
├── matrix_builder.py # Réduction SVD 
├── sample.txt # Exemple de texte 
└── graphe.png # Image de sortie générée
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install nltk scikit-learn matplotlib networkx
```

---

## Utilsiation

```bash
python main_ameliore.py sample.txt -o graphe.png -t 0.3
```

