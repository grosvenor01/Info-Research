import nltk 
import os 
from collections import Counter
import re 

print(os.listdir("Collections"))
stop_words = nltk.corpus.stopwords.words("english")



for index, filename in enumerate(os.listdir("Collections")):
    with open(f"Collections/{filename}") as file:
        doc = file.read()
        words = set(re.findall(r'\b\w+\b', doc.lower())) 
        filtered = [word for word in words if word not in stop_words]
        filtered.sort()
        with open("output.txt", "a") as file2:
            for word in filtered:
                file2.write(f"{index + 1} {word}\n")


doc2 = ""
# make all the contetn as one string 
for filename in os.listdir("Collections"):
    with open(f"Collections/{filename}", 'r') as file:
        doc2 += file.read() + "\n\n"

# get all words (without duplicates)
words = list(set(re.findall(r'\b\w+\b', doc.lower())) )
filtered = [word for word in words if word not in stop_words]

file_paths = [f"Collections/D{i}.txt" for i in range(1, 7)]
file_contents = {}

# create dec to store num file and words in it
for index, file_path in enumerate(file_paths, start=1):
    with open(file_path, 'r') as file:
        file_contents[index] = set(file.read().split())

# get a word and check in all lists
with open("outputInverse.txt", "w") as output_file:
    for word in filtered:
        found = False
        for index, contents in file_contents.items():
            if word in contents:
                output_file.write(f"{word} {index}\n")
                found = True
        if not found:
            print(f"{word} not found in any file")
