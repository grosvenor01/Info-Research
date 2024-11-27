import streamlit as st
import pandas as pd
import re
import os
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from collections import Counter
import math
import nltk
import numpy as np
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
#normal
def term_per_doc_func(processing_method, stemming_method, query , method , vect=None): 
    # Doc number , vocabulary size , taille , Term , fréqance
    #transforme the query
    doc_num = None
    try : 
        doc_num = [int(num) for num in query.split(" ")]
    except Exception as e:
        print(e)
    # Extract data 
    doc_content=""
    if method == "normal" :
        df = pd.DataFrame(columns=["Doc" , "Term" ,"Taille", "Frequance" , "Scalar"])
    else :
        df = pd.DataFrame(columns=["Term" , "Doc" ,"Taille" ,"Frequance" , "Scalar"])
    
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
                if method == "normal" :
                    df.loc[len(df)] = [i , j ,doc_size, freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition)]
                else :
                    df.loc[len(df)] = [j , i ,doc_size,freqs[j]  ,calculate_weight(j ,freqs, maximum ,apparition)]
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

            # Stemming
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
                if vect :
                    query_try = query.split(" ")
                    query_try = [stemmer.stem(i) for i in query_try]
                else:
                    query_try = stemmer.stem(query)
                
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
                if vect :
                    query_try = query.split(" ")
                    query_try = [stemmer.stem(i) for i in query_try ]
                else:
                    query_try = stemmer.stem(query)
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            else :
                stemmer =None
            # remove stop words 
            words = [word for word in words if word not in stop_words]
            doc_size = len(words)

            #Calculate freancies and max number of frequencies
            freqs , maximum = calcule_freq(words)

            #remove non needed query 
            if vect :
                words = [word for word in words if word in query_try]
            else : 
                words = [word for word in words if word == query_try]
            # Calculer l'appartition dans les autres document 
            apparition = calcule_apparition(stemmer , processing_method)
            # remove redandancy 
            words = list(set(words))
            #building df 
            for j in words :
                if method == "normal" :
                    df.loc[len(df)] = [i , j ,doc_size, freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) ]
                else :
                    df.loc[len(df)] = [j , i ,doc_size, freqs[j] ,calculate_weight(j ,freqs, maximum ,apparition) ]
        
        import numpy as np
         
        if vect:
            # produit scalaire
            df_grouped = df.groupby(["Doc"]).sum()
            df_grouped = df_grouped.sort_values(by="Scalar", ascending=False).reset_index()

            st.dataframe(df_grouped)

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
        if len(search_query.split()) > 1:
            if indexing_method == "DOCS per TERM (Inverse)":
                term_per_doc_func(processing_method, stemming_method, search_query , "inverse" , vect = True)
            else:  # TERMS per DOC
                term_per_doc_func(processing_method, stemming_method, search_query , "normal" , vect = True)