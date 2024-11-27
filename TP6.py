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


def validity_checker(query):
    query_list = query.split(" ")
    for i in range(len(query_list)-1):
        if query_list[i] not in  ["and" , "or" , "not"] and query_list[i+1] not in  ["and" , "or" , "not"]:
            return False
        elif  query_list[i] == "not" and query_list[i+1] in ["and" , "or" , "not"]:
            return False
        
    if query_list[0] in ["and" , "or" , "not"]:
        if query_list[0]  == "and" or query_list[0]  == "or" :
            return False
        elif len(query_list)==1:
            return False 

    if query_list[len(query_list)-1] in ["and" , "or" , "not"]:
        return False
    
    for i in range(1 , len(query_list)-1):
        # check if two logical comaprator mor ba3dahom 
        if query_list[i] in  ["and" , "or" , "not"] and query_list[i+1] in  ["and" , "or"]: 
            return False
    
    return True
            

st.title("Validity Checker")
st.subheader("Query:")
search_query = st.text_input("Enter your search query (document number or term):")

if st.button("Search"):
    if not search_query:
        st.error("Please enter a search query.")
    else:
        if len(search_query.split()) > 1:
           st.write(validity_checker(search_query.lower())) 
        
        elif search_query.lower() not in ["and" , "or" ,"not"]:
            st.write(True)
        
        else :
            st.write(False)
