import math
import operator
import string
import sys

# Defining constants to denote the start and end of corpus
POSITIVE_INFINITY = sys.maxsize
NEGATIVE_INFINITY = -POSITIVE_INFINITY - 1

k1 = 1.2
b = 0.75

# Stores the inverted index for the corpus
inverted_index = {}

# Stores the posting list for the terms
posting_list = {}

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


# Utility method to perform binary search for implementing next_pos
def binarysearch_high(term, low, high, current):
    while high - low > 1:
        mid = int((low + high) / 2)
        if posting_list[term][mid] <= current:
            low = mid
        else:
            high = mid
    return high


# Creates inverted index based on the corpus
def create_index(documents):
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


# Returns the next occurrence of a term given current position
def next_pos(term, current):
    # Query term not present in the inverted index
    if term not in posting_list.keys():
        return POSITIVE_INFINITY
    cache_next_pos[term] = -1
    length_posting = len(posting_list[term]) - 1
    # No next occurrence of a term
    if len(posting_list[term]) == 0 or posting_list[term][length_posting] <= current:
        return POSITIVE_INFINITY
    # Setting up cache for the first time
    if posting_list[term][0] > current:
        cache_next_pos[term] = 0
        return posting_list[term][cache_next_pos[term]]
    if cache_next_pos[term] > 0 and posting_list[cache_next_pos[term] - 1] <= current:
        low = cache_next_pos[term] - 1
    else:
        low = 0
    jump = 1
    high = low + jump
    # Galloping till the required term is passed
    while high < length_posting and posting_list[term][high] <= current:
        low = high
        jump *= 2
        high = low + jump
    if high > length_posting:
        high = length_posting
    # Peforming binary search within the limits
    cache_next_pos[term] = binarysearch_high(term, low, high, current)
    return posting_list[term][cache_next_pos[term]]


# Returns the document number given a position
def docid(position):
    if position == NEGATIVE_INFINITY:
        return NEGATIVE_INFINITY
    if position == POSITIVE_INFINITY:
        return POSITIVE_INFINITY
    doc_num = 1
    prev_length = 0
    for doc in documents:
        words = doc.split()
        doc_length = len(words) + prev_length
        if position <= doc_length:
            return doc_num
        else:
            prev_length = doc_length
            doc_num += 1
    return None


# Returns the document id of the next document containing the term
def next_doc(term, current_doc):
    search_index = doc_first_last[current_doc][1]
    pos = next_pos(term, search_index)
    doc_num = docid(pos)
    return (doc_num)


def get_idf(term):
    return float(math.log(len(documents) / inverted_index[term][-1], 2))


def doc_length(doc_id):
    return len(documents[doc_id - 1].split())


def get_tf_bm25(doc_id, term, avg_doc_length):
    for pair in inverted_index[term][:-1]:
        # If the term is present in the given document
        if pair[0] == doc_id:
            (pair[1] * (k1 + 1)) / (pair[1] + k1 * ((1 - b) + b * doc_length(doc_id) / avg_doc_length))
            return float(1 + math.log(pair[1], 2))
    return 0.0


def get_average_doc_length(documents):
    # doc_terms = documents.split();
    doc_lengths = list(map(lambda x: len(x), list(map(lambda x: x.split(), documents))))
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    return avg_doc_length


def heapify_terms(terms):
    terms = sorted(terms, key=lambda term: term["nextDoc"])
    return terms


def heapify_results(results):
    results = sorted(results, key=lambda result: result["score"])
    return results


def generate_max_scores(query):
    max_score = {}
    for term in query:
        max_score[term] = 2.2 * get_idf(term)
    max_score = sorted(max_score.items(), key=operator.itemgetter(1))
    return max_score


def rank_bm25(avg_doc_length, k, query):
    max_score = generate_max_scores(query)
    print(max_score)
    # Heap for storing the top k results
    results = []
    for i in range(k):
        results_data = {"docid": 0, "score": 0}
        results.append(results_data)
    # Heap for storing query terms
    terms = []
    for i in range(len(query)):
        terms_data = {"term": query[i], "nextDoc": next_doc(query[i], NEGATIVE_INFINITY)}
        terms.append(terms_data)
    # Establishing heap property for terms
    terms = heapify_terms(terms)
    print(terms)
    while terms[0]["nextDoc"] < POSITIVE_INFINITY:
        if results[0]["score"] > max_score[0][1]:
            terms = list(filter(lambda x: x["term"] != max_score[0][0], terms))
            del max_score[0]
            terms = heapify_terms(terms)
        d = terms[0]["nextDoc"]
        score = 0
        while terms[0]["nextDoc"] == d:
            t = terms[0]["term"]
            score = score + get_idf(t) * get_tf_bm25(d, t, avg_doc_length)
            terms[0]["nextDoc"] = next_doc(t, d)
            terms = heapify_terms(terms)
        if score > results[0]["score"]:
            results[0]["docid"] = d
            results[0]["score"] = score
            results = heapify_results(results)
    return results


def main():
    global documents
    global inverted_index
    # Reading the corpus file specified in command line
    with open(sys.argv[1], 'r') as text:
        input_string = text.read()

    # Removing punctuations and converting to lower case
    input_string = input_string.translate(str.maketrans('', '', string.punctuation)).lower()

    # Splitting the corpus into documents and separating the terms
    documents = separate_terms_in_documents(input_string)

    avg_doc_length = get_average_doc_length(documents)

    # Reading the positive query from command line
    query = normalize_query(sys.argv[3])

    # Creating an inverted index
    inverted_index = create_index(documents)


    # Displaying the top k solutions
    k = int(sys.argv[2])
    results = rank_bm25(avg_doc_length, k, query)
    print(results)


if __name__ == '__main__':
    main()
