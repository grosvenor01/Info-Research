import os
import re
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from collections import defaultdict, Counter
import nltk
import math

# Fonction pour charger le contenu d'un document
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower()

# Fonction pour obtenir les positions des mots stémmés dans un texte
def get_term_positions(text, term, stemming=None):
    positions = []
    words = re.findall(r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*', text.lower())
    
    if stemming == "Porter":
        stemmer = PorterStemmer()
    elif stemming == "Lancaster":
        stemmer = LancasterStemmer()
    else:
        stemmer = None
    
    for index, word in enumerate(words):
        processed_word = stemmer.stem(word) if stemmer else word
        if processed_word == term:
            positions.append(index + 1)
    return positions

# Fonction pour le traitement du texte (tokenization et stemming)
def process_text(text, method, stemming, stop_words):
    if method == "Split":
        terms = text.split()
    else:
        terms = re.findall(r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*', text.lower())

    if stemming == "Porter":
        stemmer = PorterStemmer()
        terms = [stemmer.stem(term) for term in terms if term.lower() not in stop_words]
    elif stemming == "Lancaster":
        stemmer = LancasterStemmer()
        terms = [stemmer.stem(term) for term in terms if term.lower() not in stop_words]
    else:
        terms = [term for term in terms if term.lower() not in stop_words]

    return terms

# Fonction pour calculer le produit scalaire pour plusieurs termes
def scalar_product(query_terms, doc_id, tf_idf_index):
    scalar_prod = 0
    for term in query_terms:
        if term in tf_idf_index and doc_id in tf_idf_index[term]:
            scalar_prod += tf_idf_index[term][doc_id]  # Poids de chaque terme dans le document
    return scalar_prod

def cosine_similarity(query_terms, doc_id, tf_idf_index):
    # Initialiser les ensembles de termes
    vocabulary = set(tf_idf_index.keys())
    
    # Initialiser les poids dans la requête (v_i) et dans le document (w_i)
    query_weights = {}
    doc_weights = {}

    # Calculer les poids pour chaque terme du vocabulaire
    for term in vocabulary:
        # Poids dans la requête : 1 si présent, sinon 0
        if term in query_terms:
            query_weights[term] = 1
        else:
            query_weights[term] = 0
        
        # Poids dans le document : TF-IDF si disponible, sinon 0
        if term in tf_idf_index and doc_id in tf_idf_index[term]:
            doc_weights[term] = tf_idf_index[term][doc_id]
        else:
            doc_weights[term] = 0

    # Calculer le produit scalaire entre v_i et w_i
    scalar_prod = 0
    for term in vocabulary:
        scalar_prod += query_weights[term] * doc_weights[term]

    # Calculer la magnitude du vecteur de la requête
    query_magnitude = 0
    for term in vocabulary:
        query_magnitude += query_weights[term] ** 2
    query_magnitude = math.sqrt(query_magnitude)

    # Calculer la magnitude du vecteur du document
    doc_magnitude = 0
    for term in vocabulary:
        doc_magnitude += doc_weights[term] ** 2
    doc_magnitude = math.sqrt(doc_magnitude)

    # Éviter la division par zéro
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0

    # Calculer la similarité cosinus
    return scalar_prod / (query_magnitude * doc_magnitude)

def jaccard_index(query_terms, doc_id, tf_idf_index):
    # Initialiser les ensembles de termes (vocabulaire)
    vocabulary = set(tf_idf_index.keys())

    # Calculer les poids dans la requête (v_i) et dans le document (w_i)
    query_weights = {}
    doc_weights = {}
    for term in vocabulary:
        # Poids dans la requête : 1 si le terme est dans query_terms, sinon 0
        query_weights[term] = 1 if term in query_terms else 0
        # Poids dans le document : TF-IDF si disponible, sinon 0
        doc_weights[term] = tf_idf_index[term][doc_id] if term in tf_idf_index and doc_id in tf_idf_index[term] else 0

    # Calculer les termes de la formule
    scalar_prod = 0  # Somme des produits v_i * w_i
    sum_query_squares = 0  # Somme des carrés v_i^2
    sum_doc_squares = 0    # Somme des carrés w_i^2

    for term in vocabulary:
        v_i = query_weights[term]
        w_i = doc_weights[term]
        scalar_prod += v_i * w_i
        sum_query_squares += v_i ** 2
        sum_doc_squares += w_i ** 2

    # Calcul du dénominateur (somme des carrés et soustraction)
    denominator = sum_query_squares + sum_doc_squares - scalar_prod

    # Éviter la division par zéro
    if denominator == 0:
        return 0

    # Calcul de l'indice de Jaccard
    return scalar_prod / denominator


# Fonction pour calculer le poids des termes
def calculate_weight(freq, max_freq, total_docs, term_doc_count):
    return (freq / max_freq) * math.log10(total_docs / term_doc_count + 1)

# Fonction pour traiter les documents
def process_documents(directory_path, method, stemming):
    stop_words = set(stopwords.words("english"))
    index_docs_per_term = defaultdict(dict)
    index_terms_per_doc = defaultdict(dict)
    doc_max_freq = {}
    doc_positions = defaultdict(lambda: defaultdict(list))
    total_docs = 0
    term_appearance_count = Counter()

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            document = load_document(file_path)
            doc_id = os.path.splitext(filename)[0]
            total_docs += 1

            terms = process_text(document, method, stemming, stop_words)
            term_freqs = Counter(terms)
            term_appearance_count.update(set(terms))

            for term in set(terms):
                doc_positions[doc_id][term] = get_term_positions(document, term, stemming)

            max_freq = max(term_freqs.values()) if term_freqs else 1
            doc_max_freq[doc_id] = max_freq
            index_terms_per_doc[doc_id] = term_freqs

            for term, freq in term_freqs.items():
                index_docs_per_term[term][doc_id] = freq

    tf_idf_index = defaultdict(dict)
    for term, docs in index_docs_per_term.items():
        doc_count_containing_term = term_appearance_count[term]
        for doc_id, freq in docs.items():
            tf_idf_index[term][doc_id] = calculate_weight(freq, doc_max_freq[doc_id], total_docs, doc_count_containing_term)

    return index_docs_per_term, index_terms_per_doc, tf_idf_index, doc_positions


def compute_bm25(query_terms, document_terms, freq, dl, avgdl, N, ni, k, b):
    score = 0
    
    # Calcul du score pour chaque terme de la requête
    for term in query_terms:
        if term in document_terms:
            term_freq = freq.get(term, 0)  # Fréquence du terme dans le document
            
            # Composant TF (Term Frequency)
            tf_component = term_freq / (k * ((1 - b) + b * (dl / avgdl)) + term_freq)
            
            # Composant IDF (Inverse Document Frequency)
            idf_component = math.log10((N - ni.get(term, 0) + 0.5) / (ni.get(term, 0) + 0.5) )
            
            # Contribution au score BM25
            score += tf_component * idf_component
    
    return score




# Interface Streamlit
st.title("Visualisation des index TF-IDF")

directory_path = st.text_input("Entrez le chemin du dossier contenant les fichiers:")
query = st.text_input("Query", "")
query_terms = query.split(" ")  # Diviser les termes de la requête

# Options de traitement
method = st.selectbox("Processing", ["Split", "RegExp"])
stemming = st.selectbox("Stemmer", ["Without", "Porter", "Lancaster"])
index_option = st.radio("Index", ["DOCS per TERM", "TERMS per DOC", "SCALAR PRODUCT", "COSINE SIMILARITY", "JACCARD INDEX","Probabolistic Model (BM25)"])

if index_option == "Probabolistic Model (BM25)" :
    # Inputs pour K et B avec Streamlit
    k = st.number_input("Entrez la valeur de K", min_value=0.0, value=1.5, step=0.1)
    b = st.number_input("Entrez la valeur de B", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

# Traitement des requêtes
if st.button("Search"):
    if os.path.isdir(directory_path):
        index_docs_per_term, index_terms_per_doc, tf_idf_index, doc_positions = process_documents(directory_path, method, stemming)

        if index_option == "DOCS per TERM":
            for term in query_terms:
                term = term.strip()
                if term:
                    stemmer = PorterStemmer() if stemming == "Porter" else LancasterStemmer() if stemming == "Lancaster" else None
                    query_term = stemmer.stem(term) if stemmer else term
                    if query_term in index_docs_per_term:
                        st.write(f"Résultats pour le terme '{query_term}' (Docs per Term):")
                        results = index_docs_per_term[query_term]
                        tf_idf_results = tf_idf_index[query_term]
                        df = pd.DataFrame({
                            "N°": list(range(1, len(results) + 1)),
                            "N°doc": list(results.keys()),
                            "Terme": [query_term] * len(results),
                            "Freq": list(results.values()),
                            "Poids": [tf_idf_results[doc_id] for doc_id in results.keys()],
                            "Positions": [doc_positions[doc_id].get(query_term, []) for doc_id in results.keys()]
                        })
                        st.dataframe(df)
                    else:
                        st.write(f"Le terme '{term}' n'a été trouvé dans aucun document.")

        elif index_option == "TERMS per DOC":
            for doc_id in query_terms:
                doc_id = doc_id.strip()
                if doc_id in index_terms_per_doc:
                    st.write(f"Résultats pour le document '{doc_id}' (Terms per Doc):")
                    results = index_terms_per_doc[doc_id]
                    sorted_terms = sorted(results.keys())
                    df = pd.DataFrame({
                        "N°": list(range(1, len(sorted_terms) + 1)),
                        "N°doc": [doc_id] * len(sorted_terms),
                        "Terme": sorted_terms,
                        "Freq": [results[term] for term in sorted_terms],
                        "Poids": [tf_idf_index[term][doc_id] for term in sorted_terms if term in tf_idf_index],
                        "Positions": [doc_positions[doc_id][term] for term in sorted_terms]
                    })
                    st.dataframe(df)
                else:
                    st.write(f"Le document '{doc_id}' n'a été trouvé ou est vide.")

        elif index_option == "SCALAR PRODUCT":
            stemmer = None
            if stemming:
                stemmer = PorterStemmer() if stemming == "Porter" else LancasterStemmer() if stemming == "Lancaster" else None
            
            # Appliquer le stemming aux termes de la requête si un stemmer est défini
            query_terms_stemmed = [stemmer.stem(term.strip()) if stemmer else term.strip() for term in query_terms if term.strip()]

            results = []
            for doc_id in index_terms_per_doc.keys():  # Boucle sur chaque document
                scalar_prod = scalar_product(query_terms_stemmed, doc_id, tf_idf_index)
                if scalar_prod != 0 :
                    results.append({"Document": doc_id, "Produit Scalaire": scalar_prod})
                
            # Afficher les résultats sous forme de tableau trié
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by="Produit Scalaire", ascending=False) 
            st.write("Produits scalaires des termes de la requête pour chaque document:")
            st.dataframe(results_df)

        elif index_option == "COSINE SIMILARITY":
            stemmer = PorterStemmer() if stemming == "Porter" else LancasterStemmer() if stemming == "Lancaster" else None
            query_terms_stemmed = [stemmer.stem(term.strip()) if stemmer else term.strip() for term in query_terms if term.strip()]

            results = []
            for doc_id in index_terms_per_doc.keys():
                cos_sim = cosine_similarity(query_terms_stemmed, doc_id, tf_idf_index)
                if cos_sim != 0 :
                    results.append({"Document": doc_id, "Similarité Cosinus": cos_sim})

            results_df = pd.DataFrame(results).sort_values(by="Similarité Cosinus", ascending=False)
            st.write("Similarité cosinus des termes de la requête pour chaque document:")
            st.dataframe(results_df)

        elif index_option == "JACCARD INDEX":
            stemmer = PorterStemmer() if stemming == "Porter" else LancasterStemmer() if stemming == "Lancaster" else None
            query_terms_stemmed = [stemmer.stem(term.strip()) if stemmer else term.strip() for term in query_terms if term.strip()]

            results = []
            for doc_id, doc_terms in index_terms_per_doc.items():
                jaccard_sim = jaccard_index(query_terms_stemmed, doc_id, tf_idf_index)
                if jaccard_sim!= 0 :
                    results.append({"Document": doc_id, "Indice de Jaccard": jaccard_sim})

            results_df = pd.DataFrame(results).sort_values(by="Indice de Jaccard", ascending=False)
            st.write("Indice de Jaccard des termes de la requête pour chaque document:")
            st.dataframe(results_df)
        
        elif index_option == "Probabolistic Model (BM25)":
            # Initialisation du Stemmer selon l'option choisie
            stemmer = PorterStemmer() if stemming == "Porter" else LancasterStemmer() if stemming == "Lancaster" else None
            query_terms_stemmed = [stemmer.stem(term.strip()) if stemmer else term.strip() for term in query_terms if term.strip()]
            


            results = []
                        # Parcours de chaque document pour calculer son score BM25
            for doc_id, document_terms in index_terms_per_doc.items():  # Assurez-vous que `index_terms_per_doc` contient {doc_id: termes du document}
                dl = sum(document_terms.values())  # Taille du document (nombre total de termes)
                freq = document_terms  # Si `document_terms` est un Counter, il contient déjà les fréquences
                avgdl = sum(sum(doc.values()) for doc in index_terms_per_doc.values()) / len(index_terms_per_doc)  # Longueur moyenne des documents
                N = len(index_terms_per_doc)  # Nombre total de documents
                ni = {
                    term: sum(1 for doc in index_terms_per_doc.values() if term in doc)
                    for term in query_terms_stemmed
                }  # Nombre de documents contenant chaque terme

                # Calcul du score BM25
                bm25_score = compute_bm25(query_terms_stemmed, document_terms, freq, dl, avgdl, N, ni, k=k, b=b)

                # Ajout du résultat
                if bm25_score != 0:
                    results.append((doc_id, bm25_score))


            # Tri des résultats par pertinence décroissante
            results = sorted(results, key=lambda x: x[1], reverse=True)

            # Affichage des résultats avec Streamlit
            st.write("### Résultats BM25 :")
            for doc_id, score in results:
                st.write(f"**Document {doc_id}** : Score = {score:.4f}")
            
    else:
        st.error("Le chemin du dossier est invalide ou vide.")