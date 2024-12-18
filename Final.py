import ipywidgets as widgets
from IPython.display import display
import nltk , math , re , os
import pandas as pd
from nltk import PorterStemmer , LancasterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
import ast 
import pandas as pd
import streamlit as st
# Initialization
Porter = nltk.PorterStemmer()
Lancaster = nltk.LancasterStemmer() 

ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*') # \d : équivalent à [0-9] 

StopWords= nltk.corpus.stopwords.words('english') 
stop_words = nltk.corpus.stopwords.words('english') 

def Extract_Regex(text):
    words = re.findall(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*", text.lower())
    return words

def Extract_Split(text):
    words = text.lower().split()
    return words

#Simplified calcule_freq - only needed for max frequency check in calculate_weight if needed
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
    weight = (freqs[word] / maximum)*math.log10(1033/apparition[word] + 1)
    return weight

def get_position(word , words):
    indices = [i+1 for i in range(len(words)) if words[i]==word]
    return indices

def term_per_doc_func(processing_method, stemming_method, query, method):
    doc_num = None
    try:
        start = int(query.split(" ")[0])
        end = int(query.split(" ")[1])
        doc_num = [num for num in range(start , end)]
    except Exception as e:
        print(e)

    filename = f"Discripteur/descripteur{processing_method.lower()}"
    if stemming_method:
        filename += stemming_method.lower()
    if method == "inverse":
        filename += "_inverse"
    filename += ".txt"

    if method == "normal":
        df = pd.DataFrame(columns=["doc", "Term", "Taille", "Frequance", "Poids", "Position"])
    else:
        df = pd.DataFrame(columns=["Term", "Doc", "Taille", "Frequance", "Poids", "Position"])

    if doc_num:
        for i in doc_num:
            print(i)
            doc_content = open(f"Collections/D{i}.txt").read()
            if processing_method == "Regex":
                words = Extract_Regex(doc_content)
            else:
                words = Extract_Split(doc_content)
            words_redandance = words
            words = [word for word in words if word not in stop_words]
            if stemming_method == "Porter":
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            elif stemming_method == "Lancaster":
                stemmer = LancasterStemmer()
                words = [stemmer.stem(word) for word in words]
                words_redandance = [stemmer.stem(word) for word in words_redandance]
            else:
                stemmer = None
            words = [word for word in words if word not in stop_words]
            doc_size = len(words)
            freqs, maximum = calcule_freq(words)
            apparition = calcule_apparition(stemmer, processing_method)
            words = list(set(words))
            for j in words:
                positions = get_position(j, words_redandance)
                if method == "normal":
                    df.loc[len(df)] = [i, j, doc_size, freqs[j], calculate_weight(j, freqs, maximum, apparition), positions]
                else:
                    df.loc[len(df)] = [j, i, doc_size, freqs[j], calculate_weight(j, freqs, maximum, apparition), positions]

    elif query:
        for i in os.listdir("Collections/"):
            doc_content = open(f"Collections/{i}").read()
            if processing_method == "Regex":
                words = Extract_Regex(doc_content)
            else:
                words = Extract_Split(doc_content)
            words_redandance = words
            words = [word for word in words if word not in stop_words]
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
            else:
                stemmer = None
            words = [word for word in words if word not in stop_words]
            doc_size = len(words)
            freqs, maximum = calcule_freq(words)
            words = [word for word in words if word == query]
            apparition = calcule_apparition(stemmer, processing_method)
            words = list(set(words))
            for j in words:
                positions = get_position(j, words_redandance)
                if method == "normal":
                    df.loc[len(df)] = [i, j, doc_size, freqs[j], calculate_weight(j, freqs, maximum, apparition), positions]
                else:
                    df.loc[len(df)] = [j, i, doc_size, freqs[j], calculate_weight(j, freqs, maximum, apparition), positions]

    try:
        df.to_csv(filename, sep='\t', index=False, header=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")
  
def rsv(query,stemming,preprocessing):
    rsv_dict = {}
    file_path = ""
    if preprocessing == "Split":
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteursplit_inverse.txt'
                    query_terms = list(set([term for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:              
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms :

                                if parts[1] in rsv_dict :                         
                                    rsv_dict[parts[1]] += float(parts[4])
                                else :
                                    rsv_dict[parts[1]] = float(parts[4])                       
                         
                case 'Porter':
                    file_path = 'Discripteur/descripteursplitporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in rsv_dict:
                                                                        
                                    rsv_dict[parts[1]] += float(parts[4])
                                else:

                                    rsv_dict[parts[1]] = float(parts[4])
                case 'Lancaster':
                    file_path = 'Discripteur/descripteursplitlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in rsv_dict:
                                                                        
                                    rsv_dict[parts[1]] += float(parts[4])
                                else:

                                    rsv_dict[parts[1]] = float(parts[4])
    else:
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteurregex_inverse.txt'
                    query_terms = list(set([term for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in rsv_dict:
                                                                        
                                    rsv_dict[parts[1]] += float(parts[4])
                                else:

                                    rsv_dict[parts[1]] = float(parts[4])
                  
                case 'Porter':
                    file_path = 'Discripteur/descripteurregexporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                            if parts[0] in  query_terms:# parts[1] is the term

                                if parts[1] in rsv_dict: # parts[2] is the document
                                                                        
                                    rsv_dict[parts[1]] += float(parts[4]) # parts[4] is the wight of the term in the document
                                else:

                                    rsv_dict[parts[1]] = float(parts[4])
                    
                   
                case 'Lancaster':
                    file_path = 'Discripteur/descripteurregexlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in rsv_dict:
                                                                        
                                    rsv_dict[parts[1]] += float(parts[4])
                                else:
                                   
                                   rsv_dict[parts[1]] = float(parts[4])

    rsv_dict = dict(sorted(rsv_dict.items(),key=lambda item:item[1],reverse=True))
    return rsv_dict

def cosine(query,stemming,preprocessing):

   
    som_wi = {}
    som_wi_carre = {}
    cosine_dict = {}
    som_vi_squred =  0
    
    
    
    file_path = ""
    if preprocessing == "Split":
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteursplit_inverse.txt'
                    query_terms = list(set([term for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:              
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteursplit.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))                                
                         
                case 'Porter':
                    file_path = 'Discripteur/descripteursplitporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteursplitporter.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            
                
                case 'Lancaster':
                    file_path = 'Discripteur/descripteursplitlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break

                    
                    file_path = 'Discripteur/descripteursplitlancaster.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            
                             
    else:
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteurregex_inverse.txt'
                    query_terms = list(set([term for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                    
                    file_path = 'Discripteur/descripteurregex.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            
                                
                case 'Porter':
                    file_path = 'Discripteur/descripteurregexporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                            if parts[0] in  query_terms:
                                # clculate the som of wi 
                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])

                                

                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
            
            
            
                                

                                
                                
                                
                    file_path = 'Discripteur/descripteurregexporter.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))
                      
                case 'Lancaster':
                    file_path = 'Discripteur/descripteurregexlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteurregexlancaster.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))    
                           

    
    print('vi',som_vi_squred)
    for doc,v in som_wi.items():
        cosine_dict[doc] =  som_wi[doc] / ( math.sqrt(som_vi_squred) * math.sqrt(som_wi_carre[doc]))
    
    cosine_dict = dict(sorted(cosine_dict.items(),key=lambda item:item[1],reverse=True))
    return cosine_dict

def jaccard(query,stemming,preprocessing):
    som_wi = {}
    som_wi_carre = {}
    jaccard_dict = {}
    som_vi_squred = 0
    file_path = ""
    if preprocessing == "Split":
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteursplit_inverse.txt'
                    query_terms = list(set([term for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:              
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])

                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteursplit.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))                                
                         
                case 'Porter':
                    file_path = 'Discripteur/descripteursplitporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                    
                    file_path = 'Discripteur/descripteursplitporter.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            
                                
                case 'Lancaster':
                    file_path = 'Discripteur/descripteursplitlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteursplitlancaster.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            
    else:
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteurregex_inverse.txt'
                    query_terms = list(set([term for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteurregex.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            
                case 'Porter':
                    file_path =  'Discripteur/descripteurregexporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                            if parts[0] in  query_terms:
                                # clculate the som of wi 
                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])
                    
                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteurregexporter.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                            

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))

                                    
                    
                    
                case 'Lancaster':
                    file_path = 'Discripteur/descripteurregexlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:

                                if parts[1] in som_wi:
                                                                        
                                    som_wi[parts[1]] += float(parts[4])
                                else:

                                    som_wi[parts[1]] = float(parts[4])

                    for term in query_terms:
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
            
                         
                            lines = file.readlines()
                            for line in lines:
                                parts = line.split()

                                if parts[0] == term:
                                    som_vi_squred += 1
                                    break
                                
                    file_path = 'Discripteur/descripteurregexlancaster.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                             # clculate the som of wi squared
                            if parts[0] in som_wi_carre:
                                                                                                           
                                    som_wi_carre[parts[0]] += (float(parts[4]) * float(parts[4]))
                            else:

                                    som_wi_carre[parts[0]] = (float(parts[4]) * float(parts[4]))            

   
    
    for doc,v in som_wi.items():
        jaccard_dict[doc] =  som_wi[doc] / ( som_vi_squred + som_wi_carre[doc] - som_wi[doc])
    
    jaccard_dict = dict(sorted(jaccard_dict.items(),key=lambda item:item[1],reverse=True))
    return jaccard_dict    

def bm25(query,stemming,preprocessing,K,B):
    dl = {}
    avdl = 0
    freq = []
    n = {}
    bm25_dict = {}
    N = 1033
    
    #initialization of the frequencies dictionnaries of each document 
    for i in range(N):
         freq.append({})
    file_path = ""
    if preprocessing == "Split":
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteursplit_inverse.txt'
                    query_terms = list(set([term for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:              
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:
                                # calculate the sum of frequncies of the term  "parts[1]" in each document "parts[2]"
                                if parts[0] not in freq[int(parts[1])-1]:
                                                                        
                                    freq[int(parts[1])-1][parts[0]] = int(parts[3])
                                

                                # calculating the number of documents in which term parts[1] appear
                                if parts[0]  in n:
                                                                        
                                    n[parts[0]] +=  1
                                
                                else:
                                    n[parts[0]] =  1
                                
                    file_path = 'Discripteur/descripteursplit.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            
                            avdl += int(parts[3])
                            

                            # calculating the sum of frequencies for each document 
                            if parts[0] in dl:
                                                                                                           
                                    dl[parts[0]] += int(parts[3])
                            else:

                                    dl[parts[0]] = int(parts[3])                                
                         
                case 'Porter':
                    file_path = 'Discripteur/descripteursplitporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:
                                # calculate the sum of frequncies of the term  "parts[1]" in each document "parts[2]"
                                if parts[0] not in freq[int(parts[1])-1]:
                                                                        
                                    freq[int(parts[1])-1][parts[0]] = int(parts[3])
                                

                                # calculating the number of documents in which term parts[1] appear
                                if parts[0]  in n:
                                                                        
                                    n[parts[0]] +=  1
                                
                                else:
                                    n[parts[0]] =  1
                    
                    file_path = 'Discripteur/descripteursplitporter.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            
                            avdl += int(parts[3])
                            

                            # calculating the sum of frequencies for each document 
                            if parts[0] in dl:
                                                                                                           
                                    dl[parts[0]] += int(parts[3])
                            else:

                                    dl[parts[0]] = int(parts[3])            
                                
                case 'Lancaster':
                    file_path = 'Discripteur/descripteursplitlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in query.split() if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:
                                # calculate the sum of frequncies of the term  "parts[1]" in each document "parts[2]"
                                if parts[0] not in freq[int(parts[1])-1]:
                                                                        
                                    freq[int(parts[1])-1][parts[0]] = int(parts[3])
                                

                                # calculating the number of documents in which term parts[1] appear
                                if parts[0]  in n:
                                                                        
                                    n[parts[0]] +=  1
                                
                                else:
                                    n[parts[0]] =  1
                                
                    file_path = 'Discripteur/descripteursplitlancaster.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            
                            avdl += int(parts[3])
                            

                            # calculating the sum of frequencies for each document 
                            if parts[0] in dl:
                                                                                                           
                                    dl[parts[0]] += int(parts[3])
                            else:

                                    dl[parts[0]] = int(parts[3])            
    else:
            match stemming:
                case 'No Stemming':
                    file_path = 'Discripteur/descripteurregex_inverse.txt'
                    query_terms = list(set([term for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:
                                # calculate the sum of frequncies of the term  "parts[1]" in each document "parts[2]"
                                if parts[0] not in freq[int(parts[1])-1]:
                                                                        
                                    freq[int(parts[1])-1][parts[0]] = int(parts[3])
                                

                                # calculating the number of documents in which term parts[1] appear
                                if parts[0]  in n:
                                                                        
                                    n[parts[0]] +=  1
                                
                                else:
                                    n[parts[0]] =  1
                                
                    file_path = 'Discripteur/descripteurregex.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            
                            avdl += int(parts[3])
                            

                            # calculating the sum of frequencies for each document 
                            if parts[0] in dl:
                                                                                                           
                                    dl[parts[0]] += int(parts[3])
                            else:

                                    dl[parts[0]] = int(parts[3])           
                case 'Porter':
                    file_path = 'Discripteur/descripteurregexporter_inverse.txt'
                    query_terms = list(set([Porter.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                    
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()

                            if parts[0] in  query_terms:
                                # calculate the sum of frequncies of the term  "parts[1]" in each document "parts[2]"
                                if parts[0] not in freq[int(parts[1])-1]:
                                                                        
                                    freq[int(parts[1])-1][parts[0]] = int(parts[3])
                                

                                # calculating the number of documents in which term parts[1] appear
                                if parts[0]  in n:
                                                                        
                                    n[parts[0]] +=  1
                                
                                else:
                                    n[parts[0]] =  1
                                     
                       
                    print(n)           
                    file_path = 'Discripteur/descripteurregexporter.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            
                            avdl += int(parts[3])
                            

                            # calculating the sum of frequencies for each document 
                            if parts[0] in dl:
                                                                                                           
                                    dl[parts[0]] += int(parts[3])
                            else:

                                    dl[parts[0]] = int(parts[2])
                    print(dl)
                                  
                    
                case 'Lancaster':
                    file_path = 'Discripteur/descripteurregexlancaster_inverse.txt'
                    query_terms = list(set([Lancaster.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords]))
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            if parts[0] in  query_terms:
                                # calculate the sum of frequncies of the term  "parts[1]" in each document "parts[2]"
                                if parts[0] not in freq[int(parts[1])-1]:
                                                                        
                                    freq[int(parts[1])-1][parts[0]] = int(parts[3])
                                

                                # calculating the number of documents in which term parts[1] appear
                                if parts[0]  in n:
                                                                        
                                    n[parts[0]] +=  1
                                
                                else:
                                    n[parts[0]] =  1
                                
                    file_path = 'Discripteur/descripteurregexlancaster.txt'
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                         
                        lines = file.readlines()
                        for line in lines:
                            parts = line.split()
                            
                            avdl += int(parts[3])
                            

                            # calculating the sum of frequencies for each document 
                            if parts[0] in dl:
                                                                                                           
                                    dl[parts[0]] += int(parts[3])
                            else:

                                    dl[parts[0]] = int(parts[3])           

    avdl /= N
    for i  in range(len(freq)):
                         sum = 0.0
                         for term, f in freq[i].items():
                              print(f'{term}:{f}')
                              sum += ( f / ( K * ( (1-B) + B * ( dl[str(i+1)] / avdl) ) + f ) ) * math.log10( ( len(freq) - n[term] + 0.5) /(n[term] + 0.5))                         
                         bm25_dict[str(i+1)] = sum
                         
    bm25_dict = dict(sorted(bm25_dict.items(),key=lambda item:item[1],reverse=True))
    return bm25_dict

def validate_logic_query(query):
    # Tokenize the query
    tokens = ExpReg.tokenize(query) 
    operators = {'AND', 'OR', 'NOT'}
    valid = True
    
    #  empty query or single invalid operator or query that ends with an operator 
    if not tokens or tokens[0] in {'AND', 'OR'} or tokens[-1] in operators:
        return False
    
  
    expect_term = True  # to expect a term or not 
    previous_token = None

    for token in tokens:
        if token in operators:
            # to Check invalid operator sequences
            if previous_token in operators:
                if token != 'NOT' or previous_token == 'NOT':
                    return False
            
            if token in {'AND', 'OR'}:
                # AND/OR must follow a term
                if expect_term:
                    return False
                expect_term = True  # After AND/OR, we expect a term
            elif token == 'NOT':
                # NOT must precede a term 
                expect_term = True

        else:
            # Token is a term
            if not expect_term:
                return False
            expect_term = False  # After a term, we expect an operator
        
        previous_token = token

    
    return valid

def boolean_model(query,stemming,preprocessing):
    boolean_dict = {}
    terms_dictionnaries = []
    N = 1033
    file_path = ""
    if preprocessing == "Split":
            match stemming:
                case 'No Stemming':
                   
                    if not validate_logic_query(query):
                        return None

                    
                    file_path = 'Discripteur/descripteursplit_inverse.txt'
                    query_terms = [term for term in query.split() if term.lower() not in StopWords]                     
                    print('Query terms:', query_terms)

                    # Extracting query terms without operators
                    query_terms_without_operators = [term for term in query_terms if term not in ['and', 'or', 'not']]
                    print('Terms without operators:', query_terms_without_operators)

                    # Initialize variables
                    operators = ['and', 'or', 'not']
                    terms_dictionaries = [[] for _ in range(N)] 

                    # looping through the terms of the query to calculate for each document the terms of the query that it contains
                    for term in query_terms_without_operators:
                         with open(file_path, 'r', encoding='utf-8') as file:
                              lines = file.readlines()
                              for line in lines:
                                   parts = line.split()
                    
                                   if parts[0] == term:
                                        doc_id = int(parts[1]) - 1  
                                        terms_dictionaries[doc_id].append(term)

                    print('Terms dictionaries:', terms_dictionaries)

                    for i in range(N):
                        query_modified = query
                        query_terms_without_stemming = [term for term in query.split() if term.lower() not in operators]
                        
                        for term in query_terms_without_stemming:
                              #print(term)
                              #print('\n')
                              # Replacing terms with 'True' or 'False' based on their presence in the document
                            
                              if term in terms_dictionaries[i]:
                                   query_modified = query_modified.replace(term, 'True')
                              else:
                                   query_modified = query_modified.replace(term, 'False')
    
                        query_modified = query_modified.replace('OR', 'or')
                        query_modified = query_modified.replace('AND', 'and')
                        query_modified = query_modified.replace('NOT', 'not')

                        print(f"Document {i + 1} Final Query for Simplification: {query_modified}")

                        

                        try:
                            boolean_dict[i] = eval(query_modified)
                        except Exception as e:
                           print(f"Error evaluating query for document {i + 1}: {e}")
                           boolean_dict[i] = False  

                    print('Boolean Dictionary:', boolean_dict)                                
                         
                case 'Porter':
                    if not validate_logic_query(query):
                        return None

                    
                    file_path = 'Discripteur/descripteursplitporter_inverse.txt'
                    query_terms = list([Porter.stem(term) for term in query.split() if term.lower() not in StopWords])                     
                    print('Query terms:', query_terms)

                    # Extracting query terms without operators
                    query_terms_without_operators = [term for term in query_terms if term not in ['and', 'or', 'not']]
                    print('Terms without operators:', query_terms_without_operators)

                    # Initialize variables
                    operators = ['and', 'or', 'not']
                    terms_dictionaries = [[] for _ in range(N)] 

                    # looping through the terms of the query to calculate for each document the terms of the query that it contains
                    for term in query_terms_without_operators:
                         with open(file_path, 'r', encoding='utf-8') as file:
                              lines = file.readlines()
                              for line in lines:
                                   parts = line.split()
                    
                                   if parts[0] == term:
                                        doc_id = int(parts[1]) - 1  
                                        terms_dictionaries[doc_id].append(term)

                    print('Terms dictionaries:', terms_dictionaries)

                    for i in range(N):
                        query_modified = query
                        query_terms_without_stemming = [term for term in query.split() if term.lower() not in operators]
                        
                        for term in query_terms_without_stemming:
                              #print(term)
                              #print('\n')
                              # Replacing terms with 'True' or 'False' based on their presence in the document
                            
                              if Porter.stem(term) in terms_dictionaries[i]:
                                   query_modified = query_modified.replace(term, 'True')
                              else:
                                   query_modified = query_modified.replace(term, 'False')
    
                        query_modified = query_modified.replace('OR', 'or')
                        query_modified = query_modified.replace('AND', 'and')
                        query_modified = query_modified.replace('NOT', 'not')

                        print(f"Document {i + 1} Final Query for Simplification: {query_modified}")

                        

                        try:
                            boolean_dict[i] = eval(query_modified)
                        except Exception as e:
                           print(f"Error evaluating query for document {i + 1}: {e}")
                           boolean_dict[i] = False  

                    print('Boolean Dictionary:', boolean_dict)            
                                
                case 'Lancaster':
                    if not validate_logic_query(query):
                        return None

                    
                    file_path = 'Discripteur/descripteursplitlancaster_inverse.txt'
                    query_terms = list([Lancaster.stem(term) for term in query.split() if term.lower() not in StopWords])                     
                    print('Query terms:', query_terms)

                    # Extracting query terms without operators
                    query_terms_without_operators = [term for term in query_terms if term not in ['and', 'or', 'not']]
                    print('Terms without operators:', query_terms_without_operators)

                    # Initialize variables
                    operators = ['and', 'or', 'not']
                    terms_dictionaries = [[] for _ in range(N)] 

                    # looping through the terms of the query to calculate for each document the terms of the query that it contains
                    for term in query_terms_without_operators:
                         with open(file_path, 'r', encoding='utf-8') as file:
                              lines = file.readlines()
                              for line in lines:
                                   parts = line.split()
                    
                                   if parts[0] == term:
                                        doc_id = int(parts[1]) - 1  
                                        terms_dictionaries[doc_id].append(term)

                    print('Terms dictionaries:', terms_dictionaries)

                    for i in range(N):
                        query_modified = query
                        query_terms_without_stemming = [term for term in query.split() if term.lower() not in operators]
                        
                        for term in query_terms_without_stemming:
                              #print(term)
                              #print('\n')
                              # Replacing terms with 'True' or 'False' based on their presence in the document
                            
                              if Lancaster.stem(term) in terms_dictionaries[i]:
                                   query_modified = query_modified.replace(term, 'True')
                              else:
                                   query_modified = query_modified.replace(term, 'False')
    
                        query_modified = query_modified.replace('OR', 'or')
                        query_modified = query_modified.replace('AND', 'and')
                        query_modified = query_modified.replace('NOT', 'not')

                        print(f"Document {i + 1} Final Query for Simplification: {query_modified}")

                        

                        try:
                            boolean_dict[i] = eval(query_modified)
                        except Exception as e:
                           print(f"Error evaluating query for document {i + 1}: {e}")
                           boolean_dict[i] = False  

                    print('Boolean Dictionary:', boolean_dict)             
    else:
            match stemming:
                case 'No Stemming':
                    if not validate_logic_query(query):
                        return None
                    file_path = 'Discripteur/descripteurregex_inverse.txt'
                    query_terms = list([term for term in ExpReg.tokenize(query) if term.lower() not in StopWords])                     
                    print('Query terms:', query_terms)

                    # Extracting query terms without operators
                    query_terms_without_operators = [term for term in query_terms if term not in ['and', 'or', 'not']]
                    print('Terms without operators:', query_terms_without_operators)

                    # Initialize variables
                    operators = ['and', 'or', 'not']
                    terms_dictionaries = [[] for _ in range(N)] 

                    # looping through the terms of the query to calculate for each document the terms of the query that it contains
                    for term in query_terms_without_operators:
                         with open(file_path, 'r', encoding='utf-8') as file:
                              lines = file.readlines()
                              for line in lines:
                                   parts = line.split()
                    
                                   if parts[0] == term:
                                        doc_id = int(parts[1]) - 1  
                                        terms_dictionaries[doc_id].append(term)

                    print('Terms dictionaries:', terms_dictionaries)

                    for i in range(N):
                        query_modified = query
                        query_terms_without_stemming = [term for term in ExpReg.tokenize(query) if term.lower() not in operators]
                        
                        for term in query_terms_without_stemming:
                              #print(term)
                              #print('\n')
                              # Replacing terms with 'True' or 'False' based on their presence in the document
                            
                              if term in terms_dictionaries[i]:
                                   query_modified = query_modified.replace(term, 'True')
                              else:
                                   query_modified = query_modified.replace(term, 'False')
    
                        query_modified = query_modified.replace('OR', 'or')
                        query_modified = query_modified.replace('AND', 'and')
                        query_modified = query_modified.replace('NOT', 'not')

                        print(f"Document {i + 1} Final Query for Simplification: {query_modified}")

                        

                        try:
                            boolean_dict[i] = eval(query_modified)
                        except Exception as e:
                           print(f"Error evaluating query for document {i + 1}: {e}")
                           boolean_dict[i] = False  

                    print('Boolean Dictionary:', boolean_dict)            
                
                case 'Porter':

                    
                    if not validate_logic_query(query):
                        return None

                    
                    file_path = 'Discripteur/descripteurregexporter_inverse.txt'
                    query_terms = list([Porter.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords])                     
                    print('Query terms:', query_terms)

                    # Extracting query terms without operators
                    query_terms_without_operators = [term for term in query_terms if term not in ['and', 'or', 'not']]
                    print('Terms without operators:', query_terms_without_operators)

                    # Initialize variables
                    operators = ['and', 'or', 'not']
                    terms_dictionaries = [[] for _ in range(N)] 

                    # looping through the terms of the query to calculate for each document the terms of the query that it contains
                    for term in query_terms_without_operators:
                         with open(file_path, 'r', encoding='utf-8') as file:
                              lines = file.readlines()
                              for line in lines:
                                   parts = line.split()
                    
                                   if parts[0] == term:
                                        doc_id = int(parts[1]) - 1  
                                        terms_dictionaries[doc_id].append(term)

                    print('Terms dictionaries:', terms_dictionaries)

                    for i in range(N):
                        query_modified = query
                        query_terms_without_stemming = [term for term in ExpReg.tokenize(query) if term.lower() not in operators]
                        
                        for term in query_terms_without_stemming:
                              #print(term)
                              #print('\n')
                              # Replacing terms with 'True' or 'False' based on their presence in the document
                            
                              if Porter.stem(term) in terms_dictionaries[i]:
                                   query_modified = query_modified.replace(term, 'True')
                              else:
                                   query_modified = query_modified.replace(term, 'False')
    
                        query_modified = query_modified.replace('OR', 'or')
                        query_modified = query_modified.replace('AND', 'and')
                        query_modified = query_modified.replace('NOT', 'not')

                        print(f"Document {i + 1} Final Query for Simplification: {query_modified}")

                        

                        try:
                            boolean_dict[i+1] = eval(query_modified)
                        except Exception as e:
                           print(f"Error evaluating query for document {i + 1}: {e}")
                           boolean_dict[i+1] = False  

                    print('Boolean Dictionary:', boolean_dict)
                
                case 'Lancaster':
                    if not validate_logic_query(query):
                        return None

                    
                    file_path = 'Discripteur/descripteurregexlancaster_inverse.txt'
                    query_terms = list([Lancaster.stem(term) for term in ExpReg.tokenize(query) if term.lower() not in StopWords])                     
                    print('Query terms:', query_terms)

                    # Extracting query terms without operators
                    query_terms_without_operators = [term for term in query_terms if term not in ['and', 'or', 'not']]
                    print('Terms without operators:', query_terms_without_operators)

                    # Initialize variables
                    operators = ['and', 'or', 'not']
                    terms_dictionaries = [[] for _ in range(N)] 

                    # looping through the terms of the query to calculate for each document the terms of the query that it contains
                    for term in query_terms_without_operators:
                         with open(file_path, 'r', encoding='utf-8') as file:
                              lines = file.readlines()
                              for line in lines:
                                   parts = line.split()
                    
                                   if parts[0] == term:
                                        doc_id = int(parts[1]) - 1  
                                        terms_dictionaries[doc_id].append(term)

                    print('Terms dictionaries:', terms_dictionaries)

                    for i in range(N):
                        query_modified = query
                        query_terms_without_stemming = [term for term in ExpReg.tokenize(query) if term.lower() not in operators]
                        
                        for term in query_terms_without_stemming:
                              #print(term)
                              #print('\n')
                              # Replacing terms with 'True' or 'False' based on their presence in the document
                            
                              if Lancaster.stem(term) in terms_dictionaries[i]:
                                   query_modified = query_modified.replace(term, 'True')
                              else:
                                   query_modified = query_modified.replace(term, 'False')
    
                        query_modified = query_modified.replace('OR', 'or')
                        query_modified = query_modified.replace('AND', 'and')
                        query_modified = query_modified.replace('NOT', 'not')

                        print(f"Document {i + 1} Final Query for Simplification: {query_modified}")

                        

                        try:
                            boolean_dict[i+1] = eval(query_modified)
                        except Exception as e:
                           print(f"Error evaluating query for document {i + 1}: {e}")
                           boolean_dict[i+1] = False  

                    print('Boolean Dictionary:', boolean_dict)           

    return boolean_dict

def show_plot(recalls, precisions):
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_title("Recall/Precision Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.plot(recalls, precisions, color='teal', marker='o')
    plt.tight_layout()
    st.pyplot(fig)

def model_evaluation(query,model_results):

    model_returned_docs = list(model_results.keys())
    real_relevant_docs = []
    query_id = 0
    print( query)
    print(model_returned_docs)
    file_path = 'Converters/MED.QRY'
    query_id = None

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_id = None
        current_query = ""
    
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace

            # Check if the line starts a new query with .I
            if line.startswith(".I"):
                # If a query has been accumulated and matches the input query, set the query_id
                if current_query.strip() == query:
                    query_id = current_id
                    break
            
                # Start a new query
                current_id = int(line.split()[1])  # Extract the query ID
                current_query = ""  # Reset the query text for the new query
        
                # Accumulate query text, skip lines starting with .W
            elif not line.startswith(".W"):
                current_query += line + " "  # Add the line to the query text

            # Final check in case the matching query is at the end of the file
            if current_query.strip() == query:
                query_id = current_id

         # Output the result
    if query_id is not None:
        print(f"The query ID is: {query_id}")
 
    else:
        print("Query not found.")
        return None , None , None , None ,  None , None , None , None , None
    file_path = 'Converters/MED.REL'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if int(parts[0]) == query_id:
                real_relevant_docs.append(parts[2])
    
    print(real_relevant_docs)
    
    selected_relevant_docs = 0
    selected_relevant_docs_rang_5 = 0
    selected_relevant_docs_rang_10 = 0

    for i,doc in enumerate(model_returned_docs):

        if i < 5:
            if doc in real_relevant_docs:
                selected_relevant_docs_rang_5 += 1
                selected_relevant_docs_rang_10 += 1
                selected_relevant_docs += 1
        elif i < 10:
            if doc in real_relevant_docs:
                selected_relevant_docs_rang_10 += 1
                selected_relevant_docs += 1
        else:
            if doc in real_relevant_docs:
                selected_relevant_docs += 1
    
    print('selected_relevant_docs:',selected_relevant_docs)
    p = selected_relevant_docs / len(model_returned_docs)

    p5 = selected_relevant_docs_rang_5 / 5

    p10 = selected_relevant_docs_rang_10 / 10

    r = selected_relevant_docs / len(real_relevant_docs)

    F_score = ( 2 * p * r ) / ( p + r)

    recalls = []
    precesions = []
    selected_relevant_docs = 0
    print('len(real_relevant_docs)',len(real_relevant_docs))
    for i,doc in enumerate(model_returned_docs):
            if doc in real_relevant_docs:
                selected_relevant_docs += 1
            print('i + 1 ', i + 1 )
            print('selected_relevant_docs' , selected_relevant_docs)
            p1 = selected_relevant_docs / ( i + 1 )
            r1 = selected_relevant_docs / len(real_relevant_docs)
            precesions.append(p1)
            recalls.append(r1)
    recalls2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8,0.9,1.0]
    precisions2 = []
    for r2 in recalls2:
        max_precision = 0
        for rec, pre in zip(recalls, precesions):
                if rec >= r2:
                    max_precision = max(max_precision, pre)
        precisions2.append(max_precision)
    return p , p5 , p10 , r ,  F_score , precesions , recalls , precisions2 , recalls2

def affichage_simple(query, stemming, preprocessing, method):
    filename = f"Discripteur/descripteur{preprocessing.lower()}"
    if stemming and stemming != "No Stemming":
        filename += stemming.lower()

    if method == "inverse":
        filename += "_inverse"
    filename += ".txt"

    if stemming == "Porter":
        stemmer = PorterStemmer()
        query = [stemmer.stem(word) for word in query.split(" ")]
    elif stemming == "Lancaster":
        stemmer = LancasterStemmer()
        query = [stemmer.stem(word) for word in query.split(" ")]
    try:
        # Read the file line by line
        data = []
        with open(filename, "r") as file:
            for line in file:
                # Split the line by whitespace
                parts = line.strip().split()
                
                # Use only the first 5 elements
                if len(parts) >= 5:
                    word, col1, col2, col3, score = parts[:5]
                    positions = " ".join(parts[5:])  
                    if word in query : 
                        data.append({
                            "Word": word,
                            "Col1": int(col1),
                            "Col2": int(col2),
                            "Col3": int(col3),
                            "Score": float(score),
                            "positions": positions
                        })

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame in Streamlit
        st.subheader("Processed Data")
        st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")



        
"""term_per_doc_func("Split", "Porter", "1 1033", "normal")
term_per_doc_func("Split", "Lancaster", "1 1033", "normal")
term_per_doc_func("Split", None, "1 1033", "normal")
term_per_doc_func("Split", "Porter", "1 1033", "inverse")
term_per_doc_func("Split", "Lancaster", "1 1033", "inverse")
term_per_doc_func("Split", None, "1 1033", "inverse")
term_per_doc_func("Regex", "Porter", "1 1033", "normal")
term_per_doc_func("Regex", "Lancaster", "1 1033", "normal")
term_per_doc_func("Regex", None, "1 1033", "normal")
term_per_doc_func("Regex", "Porter", "1 1033", "inverse")
term_per_doc_func("Regex", "Lancaster", "1 1033", "inverse")
term_per_doc_func("Regex", None, "1 1033", "inverse")
"""
