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
            words_ = Extract_Regex(text)
        else : 
            words_ = Extract_Split(text)
        if stemmer:
            words_ = [stemmer.stem(word) for word in words_]
        words_ = [word for word in words_ if word not in stop_words]
        words_ = list(set(words_))
        all_words.extend(words_)
    
    apparition = Counter(all_words)
    return apparition 

def calculate_weight(word ,freqs, maximum , apparition):
    weight = (freqs[word] / maximum)*math.log10(6/apparition[word] + 1)
    return weight

def get_position(word , words):
    indices = [i+1 for i in range(len(words)) if words[i]==word]
    return indices
#normal
def term_per_doc_func(processing_method, stemming_method, query , method): 
    # Doc number , vocabulary size , taille , Term , fréqance
    # transforme the query
    doc_num = None
    try : 
        doc_num = [int(num) for num in query.split(" ")]
    except Exception as e:
        print(e)
    # Extract data 
    doc_content=""
    if method == "normal" :
        df = pd.DataFrame(columns=["doc" , "Term" ,"Taille", "Frequance" , "Poids" , "Position"])
    else :
        df = pd.DataFrame(columns=["Term" , "Doc" ,"Taille" ,"Frequance" , "Poids" , "Position"])
    if doc_num :
        for i in doc_num:
            doc_content = open(f"Collections/D{i}.txt").read()

            #Tokenization
            if processing_method == "Regex":
                words = Extract_Regex(doc_content)
            else : 
                words = Extract_Split(doc_content)
            
            #Saved for Positions 
            words_redandance = words 

            # remove stop words 
            words = [word for word in words if word not in stop_words]

            #Stemming
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            else :
                stemmer =None
            
            # remove stop words second time 
            words = [word for word in words if word not in stop_words]
            
            #saved for doc length 
            doc_size = len(words)

            #Calculate freancies and max number of frequencies
            freqs , maximum = calcule_freq(words)

            # Calculer l'appartition dans les autres document 
            apparition = calcule_apparition(stemmer , processing_method)

            # remove redandancy 
            words = list(set(words))

            #building df 
            for j in words :
                positions = get_position(j , words_redandance)
                if method == "normal" :
                    df.loc[len(df)] = [i , j ,doc_size, freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) , positions]
                else :
                    df.loc[len(df)] = [j , i ,doc_size,freqs[j]  ,calculate_weight(j ,freqs, maximum ,apparition) , positions]
        if method == "normal":
            display_results(df)
        else :
            display_results(df.sort_values(by="Term").reset_index())

    elif query :
        for i in os.listdir("Collections/"):
            doc_content = open(f"Collections/{i}").read()
            #Tokenization
            if processing_method == "Regex":
                words = Extract_Regex(doc_content)
            else : 
                words = Extract_Split(doc_content)
            
            #Saved for Positions 
            words_redandance = words 

            # remove stop words 
            words = [word for word in words if word not in stop_words]

            #Stemming
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
                query = stemmer.stem(query)
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
                query = stemmer.stem(query)
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            else :
                stemmer =None
            
            # remove stop words 
            words = [word for word in words if word not in stop_words]

            doc_size = len(words)

            #Calculate freancies and max number of frequencies
            freqs , maximum = calcule_freq(words)

            #remove non needed query 
            words = [word for word in words if word == query]

            # Calculer l'appartition dans les autres document 
            apparition = calcule_apparition(stemmer , processing_method)

            # remove redandancy 
            words = list(set(words))
            #building df 
            for j in words :
                positions = get_position(j , words_redandance)
                if method == "normal" :
                    df.loc[len(df)] = [i , j ,doc_size, freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) , positions]
                else :
                    df.loc[len(df)] = [j , i ,doc_size, freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) , positions]
        if method == "normal":
            display_results(df)
        else :
            display_results(df.sort_values(by="Term").reset_index())

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
        if indexing_method == "DOCS per TERM (Inverse)":
            term_per_doc_func(processing_method, stemming_method, search_query , "inverse")
        else:  # TERMS per DOC
            term_per_doc_func(processing_method, stemming_method, search_query , "normal")