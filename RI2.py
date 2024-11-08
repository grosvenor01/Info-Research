import streamlit as st
import pandas as pd
import re
import os
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from collections import Counter
import math
import nltk

stop_words = list(set(stopwords.words('english')))
def display_results(results_df):
    st.write("Results (Terms per Document):")
    st.dataframe(pd.DataFrame(results_df).style.set_properties(**{'width': '800px'}))

def Extract_Regex(text):
    pattern = '(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*'
    words = re.findall(pattern , text)
    return words

def Extract_Split(text):
    words = text.split()
    return words 

def calcule_freq(words):
    freqs = Counter(words)
    maximum=0
    for key in freqs:
        maximum = max(freqs[key] , maximum)
    return freqs , maximum


def calcule_apparition(stemmer , tokenizer):
    all_words=[]
    for i in os.listdir("Collections/"):
        text = open(f"Collections/{i}").read()
        if tokenizer =="Regex": 
            words = Extract_Regex(text)
        else : 
            words = Extract_Split(text)

        words = [stemmer.stem(word) for word in words]
        words = [word for word in words if word not in stop_words]
        words = list(set(words))
        all_words.extend(words)
    
    apparition = Counter(all_words)
    return apparition

def calculate_weight(word ,freqs, maximum , apparition):
    weight = (freqs[word] / maximum)*math.log10(6/apparition[word] + 1)
    return weight

def get_position(word , words):
    indices = [i+1 for i in range(len(words)) if words[i]==word]
    return indices
#normal
def term_per_doc_func(processing_method, stemming_method, query): 
    # Doc number , vocabulary size , taille , Term , fréqance

    #transforme the query
    doc_num = None
    try : 
        doc_num = [int(num) for num in query.split(" ")]
    except Exception as e:
        print(e)
    
    # Extract data 
    doc_content=""
    df = pd.DataFrame(columns=["doc" , "Term" , "Frequance" , "Poids" , "Position" , "just a spacer"])
    if doc_num :
        for i in doc_num:
            doc_content += open(f"Collections/D{i}.txt").read()
            #Tokenization
            if processing_method == "Regex":
                words = Extract_Regex(doc_content)
            else : 
                words = Extract_Split(doc_content)
            #Stemming
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
            
            # final repeated version 
            words_redandance = words 

            # remove stop words 
            words = [word for word in words if word not in stop_words]

            #Calculate freancies and max number of frequencies
            freqs , maximum = calcule_freq(words)

            # Calculer l'appartition dans les autres document 
            apparition = calcule_apparition(stemmer , processing_method)

            # remove redandancy 
            words = list(set(words))

            #building df 
            for j in words :
                positions = get_position(j , words_redandance)
                df.loc[len(df)] = [i , j , freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) , positions , "     "]
        display_results(df)
    elif query :
        for i in os.listdir("Collections/"):
            doc_content = open(f"Collections/{i}").read()
            #Tokenization
            if processing_method == "Regex":
                words = Extract_Regex(doc_content)
            else : 
                words = Extract_Split(doc_content)
            
            #Stemming
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
                query = stemmer.stem(query)
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
                query = stemmer.stem(query)
            
            # final repeated version 
            words_redandance = words 

            # remove stop words 
            words = [word for word in words if word not in stop_words]

            #remove non needed query 
            words = [word for word in words if word == query]

            #Calculate freancies and max number of frequencies
            freqs , maximum = calcule_freq(words)

            # Calculer l'appartition dans les autres document 
            apparition = calcule_apparition(stemmer , processing_method)

            # remove redandancy 
            words = list(set(words))
            #building df 
            for j in words :
                positions = get_position(j , words_redandance)
                df.loc[len(df)] = [i , j , freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) , positions , "     "]
        display_results(df)
    else :
        print("specify your query")
    pass
# inverse
def doc_per_term_func(processing_method, stemming_method, query):
    pass


st.title("Search and Indexing Tool")

st.subheader("Query:")
search_query = st.text_input("Enter your search query (document number or term):")

st.subheader("Processing Options")
processing_method = st.selectbox("Tokenization:", ["Regex","Split"])
stemming_method = st.selectbox("Stemmer:", ["Porter", "Without", "Lancaster"])

st.subheader("Indexing Options")
indexing_method = st.radio("Index Type:", ["DOCS per TERM (Inverse)", "TERMS per DOC (Normale)"])

if st.button("Search"):
    if not search_query:
        st.error("Please enter a search query.")
    else:
        if indexing_method == "DOCS per TERM":
            doc_per_term_func(processing_method, stemming_method, search_query)
        else:  # TERMS per DOC
            term_per_doc_func(processing_method, stemming_method, search_query)