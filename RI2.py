import streamlit as st
import pandas as pd
import re
import os
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from collections import Counter
import math

import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def display_results(results_df):
    st.write("Results (Terms per Document):")
    st.dataframe(pd.DataFrame(results_df).style.set_properties(**{'width': '800px'}))

def display_results2(results_df):
    st.write("Results (Documents per Term):")
    st.dataframe(pd.DataFrame(results_df).style.set_properties(**{'width': '800px'}))

def term_per_doc_func(processing_method, stemming_method, doc_num=None, query=None): 
    collections_path = "C:\\Users\\abdo7\\OneDrive\\Bureau\\RI\\venv\\RI\\Collections"
    regex = r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*'
    doc=""
    for index, filename in enumerate(os.listdir(collections_path)):
        with open(f"Collections/{filename}") as file:
            document = file.read()
            #Tokenization
            if processing_method == "Split" : 
                words = document.split()
            elif processing_method == "Regex":
                words = re.findall(regex, document.lower())
            #Stemmer
            stemmer = None
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
            #All in one doc
            doc += " ".join(list(set(words)))

    all_data=[]
    apparition = Counter(re.findall(regex, doc.lower())) 
    for index, filename in enumerate(os.listdir(collections_path)):
        with open(f"Collections/{filename}") as file:   
            doc = file.read()
            
            words = re.findall(regex, doc.lower())
            if query : 
                filtered = [word for word in words if word not in stop_words and word==query]
            else : 
                filtered = [word for word in words if word not in stop_words and filename==f"D{doc_num}.txt"]
            if stemmer : 
                filtered = [stemmer.stem(word) for word in filtered]
            freqs = Counter(filtered)
            print("===========len=======:" + str(len(filtered)))
            filtered = list(set(filtered))
            maximum = 0
            for key in freqs :
                maximum = max(freqs[key] , maximum)  
            for word in filtered:
                try : 
                    poids = (freqs[word]/maximum)*((math.log10(6)/apparition[word])+1)
                    all_data.append({"index" : index + 1 , "word" :word , "freq" : freqs[word] , "poids" :poids})
                except Exception as e: 
                    pass
    display_results(all_data)

def doc_per_term_func(processing_method, stemming_method, doc_num=None, query=None):
    """
    Processes documents and calculates term frequencies and weights per term, showing document occurrences.
    """
    collections_path = "C:\\Users\\abdo7\\OneDrive\\Bureau\\RI\\venv\\RI\\Collections"
    regex = r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*"

    stemmer = None
    if stemming_method == "Porter":
        stemmer = PorterStemmer()
    elif stemming_method == "Lancaster":
        stemmer = LancasterStemmer()

    term_data = {}  
    for index, filename in enumerate(os.listdir(collections_path)):
        filepath = os.path.join(collections_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                document = file.read()
                if processing_method == "Split":
                    words = document.split()
                elif processing_method == "Regex":
                    words = re.findall(regex, document.lower())
                else:
                    raise ValueError("Invalid processing_method. Choose 'Split' or 'Regex'.")

                if stemmer:
                    words = [stemmer.stem(word) for word in words]
                
                if doc_num:
                    words = [word for word in words if word not in stop_words and filename==f"D{doc_num}.txt"]
                else : 
                    words = [word for word in words if word not in stop_words and word==query]
                print("===========len=======:"+str(len(words)))
                words = list(set(words))
                for word in words:
                    if word not in term_data:
                        term_data[word] = {'docs': [], 'total_freq': 0}
                    term_data[word]['docs'].append(index + 1)
                    term_data[word]['total_freq'] += 1

        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error processing file {filename}: {e}")
            continue

    #Calculate weights (this part needs adjustment based on your desired weighting scheme)
    all_data = []
    total_docs = len(os.listdir(collections_path))
    for term, data in term_data.items():
        idf = math.log10(total_docs / len(data['docs']))  #Inverse Document Frequency
        for doc_num in data['docs']:
            tf = data['total_freq'] / total_docs #Term Frequency (simplified for this example)
            weight = tf * idf
            all_data.append({"term": term, "doc_num": doc_num, "freq": data['total_freq'], "weight": weight})

    display_results2(all_data)


st.title("Search and Indexing Tool")

st.subheader("Query:")
search_query = st.text_input("Enter your search query (document number or term):")

st.subheader("Processing Options")
processing_method = st.selectbox("Tokenization:", ["Regex","Split"])
stemming_method = st.selectbox("Stemmer:", ["Porter", "Without", "Lancaster"])

st.subheader("Indexing Options")
indexing_method = st.radio("Index Type:", ["DOCS per TERM", "TERMS per DOC"])

if st.button("Search"):
    if not search_query:
        st.error("Please enter a search query.")
    else:
        try:
            if indexing_method == "DOCS per TERM":
                try:
                    doc_num = int(search_query)
                    doc_per_term_func(processing_method, stemming_method, doc_num=doc_num)
                except ValueError:
                    doc_per_term_func(processing_method, stemming_method, query=search_query)
            else:  # TERMS per DOC
                try:
                    doc_num = int(search_query)
                    term_per_doc_func(processing_method, stemming_method, doc_num=doc_num)
                except ValueError:
                    term_per_doc_func(processing_method, stemming_method, query=search_query)
        except Exception as e:
            st.exception(e)