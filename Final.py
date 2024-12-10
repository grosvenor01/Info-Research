import ipywidgets as widgets
from IPython.display import display
import nltk , math , re , os
import pandas as pd
from nltk import PorterStemmer , LancasterStemmer
from collections import Counter
# Initialization
Porter = nltk.PorterStemmer()
Lancaster = nltk.LancasterStemmer() 

ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*') # \d : équivalent à [0-9] 
StopWords = nltk.corpus.stopwords.words('english') 
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
    weight = (freqs[word] / maximum)*math.log10(6/apparition[word] + 1)
    return weight

def get_position(word , words):
    indices = [i+1 for i in range(len(words)) if words[i]==word]
    return indices

def term_per_doc_func(processing_method, stemming_method, query, method):
    doc_num = None
    try:
        doc_num = [int(num) for num in query.split(" ")]
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
                                
                    file_path = 'Discripteur/ddescripteurregexlancaster.txt'
                    
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
    N = 6
    
    #initialization of the frequencies dictionnaries of each document 
    for i in range(6):
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
    N = 6
    file_path = ""
    if preprocessing == "Split":
            match stemming:
                case 'No Stemming':
                   
                    if not validate_logic_query(query):
                        return None

                    
                    file_path = 'Discripteur/descripteursplit_inverse.txt'
                    query_terms = list([term for term in query.split() if term.lower() not in StopWords])                     
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

"""term_per_doc_func("Split", "Porter", "1 2 3 4 5 6", "normal")
term_per_doc_func("Split", "Lancaster", "1 2 3 4 5 6", "normal")
term_per_doc_func("Split", None, "1 2 3 4 5 6", "normal")
term_per_doc_func("Split", "Porter", "1 2 3 4 5 6", "inverse")
term_per_doc_func("Split", "Lancaster", "1 2 3 4 5 6", "inverse")
term_per_doc_func("Split", None, "1 2 3 4 5 6", "inverse")

term_per_doc_func("Regex", "Porter", "1 2 3 4 5 6", "normal")
term_per_doc_func("Regex", "Lancaster", "1 2 3 4 5 6", "normal")
term_per_doc_func("Regex", None, "1 2 3 4 5 6", "normal")
term_per_doc_func("Regex", "Porter", "1 2 3 4 5 6", "inverse")
term_per_doc_func("Regex", "Lancaster", "1 2 3 4 5 6", "inverse")
term_per_doc_func("Regex", None, "1 2 3 4 5 6", "inverse")"""

print(bm25('LLM-based solutions for information retrieval','Porter','Reg',1.50,0.75))