with open("MED.ALL" , "r") as file :
    text = file.read()

list_of_docs = text.split(".W")
list_of_docs = list_of_docs[1:]
for index , i in enumerate(list_of_docs):
    with open(f"Collections/D{index+1}.txt" , "w+") as file :
        file.write(i[:-5])