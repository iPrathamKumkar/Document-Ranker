import sys
import string

def create_index(documents):
    inverted_index = {}
    current_doc = 1
    current_pos = 1
    for line in documents:
        if(line==''):
            current_doc += 1
            current_pos = 1
        else:
            for word in line.split():
                if word not in inverted_index:
                    inverted_index[word]=[[current_doc,1,[current_pos]]]
                else:
                    posting_list=inverted_index[word]
                    if (posting_list[-1][0]!=current_doc):
                       posting_list+=[[current_doc,1,[current_pos]]]
                    else:
                       posting_list[-1][1]+=1
                       posting_list[-1][2] += [current_pos]
                current_pos+=1
    number_of_docs=current_doc
    print(inverted_index)
    return inverted_index

# num_results = int(sys.argv[2])
# query = str(sys.argv[3])
token_frequency=[]
term_frquency=[]
with open(sys.argv[1],'r') as f:
    collection=f.read()
corpus=collection.translate(collection.maketrans('','',string.punctuation)).lower()
documents=corpus.split('\n')
#print(documents)
Inverted_index=create_index(documents)

#print(Inverted_index)

