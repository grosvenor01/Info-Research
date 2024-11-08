import nltk 
import os  , re
from collections import Counter
import math 

stop_words = nltk.corpus.stopwords.words("english")
doc2=""
for index, filename in enumerate(os.listdir("Collections")):
    with open(f"Collections/{filename}") as file:
        document = file.read()
        words = re.findall(r"\w+(?:[-']\w+)*\b|[A-Z][a-z]*(?:\.[a-z]+)?\b|[a-zA-Z]+(?:\.[a-zA-Z]+)*", document.lower())
        print(words)
        doc2 += " ".join(list(set(words)))
apparition = Counter(re.findall(r"\w+(?:[-']\w+)*\b|[A-Z][a-z]*(?:\.[a-z]+)?\b|[a-zA-Z]+(?:\.[a-zA-Z]+)*", doc2.lower()))

for index, filename in enumerate(os.listdir("Collections")):
    with open(f"Collections/{filename}") as file:
        doc = file.read()
        words = re.findall(r"\w+(?:[-']\w+)*\b|[A-Z][a-z]*(?:\.[a-z]+)?\b|[a-zA-Z]+(?:\.[a-zA-Z]+)*", doc.lower())
        filtered = [word for word in words if word not in stop_words]
        freqs = Counter(filtered)

        words = list(set(words))
        filtered = [word for word in words if word not in stop_words]
        maximum = 0
        for key in freqs :
            maximum = max(freqs[key] , maximum)  
        
        with open("freq.txt", "a") as file2:
            for word in filtered:
                print(word)
                try : 
                    poids = (freqs[word]/maximum) * math.log10(6/apparition[word]+1)
                except Exception as e: 
                    print("")
                if index != 0 :
                    file2.write(f"{index} {word} {freqs[word]} {poids}\n")