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

def build_formula(formula):
    stack = []
    operators = {"not": 1, "and": 2, "or": 3} 

    for token in formula:
        if token in operators:
            if token == "not":
                if not stack:
                    return None 
                operand = stack.pop()
                stack.append(not operand)
            else: # "and" or "or"
                if len(stack) < 2:
                    return None
                operand2 = stack.pop()
                operand1 = stack.pop()
                if token == "and":
                    stack.append(operand1 and operand2)
                else:  # "or"
                    stack.append(operand1 or operand2)
        elif isinstance(token, bool):
            stack.append(token)
        elif isinstance(token, str):
            try:
                stack.append(eval(token)) 
            except NameError:
                return None 

        else:
            return None 

    if len(stack) != 1:
        return None 
    print("dady")
    print(stack[0])

#normal
st.title("Validity Checker")
st.subheader("Query:")
search_query = st.text_input("Enter your search query (document number or term):")
if st.button("Search"):
    if not search_query:
        st.error("Please enter a search query.")
    else:
        df = pd.DataFrame(columns=["doc" , "Term" ,"Taille", "Frequance" , "Poids" , "Position"])
        query = search_query.split(" ")
        search_query.split()
        st.write(f"Validity : {validity_checker(search_query.lower())}")
        if len(search_query) > 1:
            if validity_checker(search_query.lower()):
                build_formula(query)
                for i in os.listdir("Collections/"):
                    doc_content = open(f"Collections/{i}").read()
                    # Tokenization
                    words = Extract_Regex(doc_content)
                    # Saved for Positions 
                    words_redandance = words 
                    # 
                    st.write(words.__contains__(query))
        
        elif search_query.lower() not in ["and" , "or" ,"not"]:
            st.write(True)
        
        else :
            st.write(False)
