import os
import re
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from collections import defaultdict, Counter
import nltk
import math


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
# Fonction pour charger le contenu d'un document
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower()
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

def compute_bm25(query_terms, document_terms, freq, dl, avgdl, N, ni, k=1.5, b=0.75):
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
# Fonction pour calculer le poids des termes

def calculate_weight(freq, max_freq, total_docs, term_doc_count):
    return (freq / max_freq) * math.log10(total_docs / term_doc_count + 1)

st.title("Visualisation des index TF-IDF")

directory_path = st.text_input("Entrez le chemin du dossier contenant les fichiers:")
query = st.text_input("Query", "")
query_terms = query.split(" ")

# Options de traitement
method = st.selectbox("Processing", ["Split", "RegExp"])
stemming = st.selectbox("Stemmer", ["Without", "Porter", "Lancaster"])

# Traitement des requêtes
if st.button("Search"):
    if os.path.isdir(directory_path):
        index_docs_per_term, index_terms_per_doc, tf_idf_index, doc_positions = process_documents(directory_path, method, stemming)
        stemmer = PorterStemmer() if stemming == "Porter" else LancasterStemmer() if stemming == "Lancaster" else None
        query_terms_stemmed = [stemmer.stem(term.strip()) if stemmer else term.strip() for term in query_terms if term.strip()]

        results = []
        for doc_id, document_terms in index_terms_per_doc.items():
            dl = sum(document_terms.values())
            freq = document_terms
            avgdl = sum(sum(doc.values()) for doc in index_terms_per_doc.values()) / len(index_terms_per_doc)
            N = len(index_terms_per_doc)
            ni = {
                term: sum(1 for doc in index_terms_per_doc.values() if term in doc)
                for term in query_terms_stemmed
            }
            bm25_score = compute_bm25(query_terms_stemmed, document_terms, freq, dl, avgdl, N, ni)
            results.append((doc_id, bm25_score))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        st.write("### Résultats BM25 :")
        if results: # Check if results are not empty
            # Create a DataFrame for better visualization
            results_df = pd.DataFrame(results, columns=['Document ID', 'BM25 Score'])
            st.dataframe(results_df) # Highlight the highest score
        else:
            st.write("Aucun résultat trouvé.")

    else:
        st.error("Le chemin du dossier est invalide ou vide.")