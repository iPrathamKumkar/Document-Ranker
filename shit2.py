import sys
import string

def create_index(documents):
    Inverted_index={}
    current_pos=1
    for line in documents:
        if(line==''):
            continue
        else:
            for word in line.split():
                if word not in Inverted_index:
                    Inverted_index[word]=[1,[current_pos]]
                else:
                    posting_list=Inverted_index[word]
                    posting_list[0]+=1
                    posting_list[1] += [current_pos]
                current_pos+=1
    print(Inverted_index)
    return Inverted_index

num_results = int(sys.argv[2])
query = str(sys.argv[3])
token_frequency=[]
term_frquency=[]
with open(sys.argv[1],'r') as f:
    collection=f.read()
corpus=collection.translate(collection.maketrans('','',string.punctuation)).lower()
documents=corpus.split('\n')
#print(documents)
Inverted_index=create_index(documents)

#print(Inverted_index)

