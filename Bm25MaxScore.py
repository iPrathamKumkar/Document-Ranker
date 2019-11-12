import string
import sys

# Defining constants to denote the start and end of corpus
POSITIVE_INFINITY = sys.maxsize
NEGATIVE_INFINITY = -POSITIVE_INFINITY - 1

# Utility method to separate the terms in a document
from functools import reduce

# Cache used in galloping search for next position
cache_next_pos = {}

# Cache used in galloping search for previous position
cache_prev_pos = {}

# Stores the starting and ending position for each document
doc_first_last = {}


def separate_terms_in_documents(input_string):
    docs = input_string.split('\n\n')
    for i in range(len(docs)):
        docs[i] = docs[i].replace('\n', ' ')
        return docs


# Converts query terms to lower case
def normalize_query(query):
    tmp_query = []
    for x in query.split():
        # Stripping punctuations from query terms and converting to lowercase
        tmp_query.append(x.translate(str.maketrans('', '', string.punctuation)).lower())
    # query = reduce(lambda x, y: x + ' ' + y, tmp_query)
    return tmp_query


# Utility method to create posting lists
def create_posting(documents):
    # Stores the posting list for the terms
    posting_list = {}

    current_pos = 1
    for doc in documents:
        for word in doc.split():
            if word not in posting_list:
                posting_list[word] = [current_pos]
            else:
                x = posting_list[word]
                x += [current_pos]
            current_pos += 1
    return posting_list


# Utility method to store document boundaries
def store_first_last_doc(documents):
    prev_length = 0
    for i in range(1, len(documents) + 1):
        words = documents[i - 1].split()
        doc_length = prev_length + len(words)
        doc_first_last[i] = (prev_length + 1, doc_length)
        prev_length = doc_length
    doc_first_last[NEGATIVE_INFINITY] = (NEGATIVE_INFINITY, 0)
    doc_first_last[POSITIVE_INFINITY] = (doc_first_last[len(documents)][1] + 1, POSITIVE_INFINITY)
    return doc_first_last


# Creates inverted index based on the corpus
def create_index(documents):
    # Stores the inverted index for the corpus
    inverted_index = {}
    current_doc = 1
    total_count = {}
    for doc in documents:
        appeared = []
        # Storing doc_id, doc_count for each term
        for word in doc.split():
            # First occurrence of the term
            if word not in inverted_index:
                inverted_index[word] = [[current_doc, 1]]
                appeared.append(word)
                total_count[word] = 1
            # First occurrence in a particular document
            elif word not in appeared:
                inverted_index[word].append([current_doc, 1])
                appeared.append(word)
                total_count[word] += 1
            # If word has already appeared in the document
            else:
                inverted_index[word][-1][1] += 1
        current_doc += 1
    # Storing the document frequency for each term
    for word in total_count.keys():
        inverted_index[word].append(total_count[word])
    create_posting(documents)
    store_first_last_doc(documents)
    return inverted_index


def main():
    # Reading the corpus file specified in command line
    with open(sys.argv[1], 'r') as text:
        input_string = text.read()

    # Removing punctuations and converting to lower case
    input_string = input_string.translate(str.maketrans('', '', string.punctuation)).lower()

    # Splitting the corpus into documents and separating the terms
    documents = separate_terms_in_documents(input_string)

    # Reading the positive query from command line
    query = normalize_query(sys.argv[3])

    # Creating an inverted index
    inverted_index = create_index(documents)

    # Generating a set of documents satisfying the given query

    # Displaying the top k solutions
    k = int(sys.argv[2])

    print(inverted_index)
    print(query)


if __name__ == '__main__':
    main()
