# Importing required libraries
import sys
import string
import math

# Defining the start and end of the corpus
from functools import reduce

POSITIVE_INFINITY = sys.maxsize
NEGATIVE_INFINITY = -POSITIVE_INFINITY - 1

# Storing boolean operators as constants
AND = '_AND'
OR = '_OR'

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

# Stores the set of documents satisfying the positive query
valid_docs = []

# Stores the results of the VSM computations
result = {}


# Defining an expression tree class to represent the positive boolean query
class ExpressionTree:
    def __init__(self, left, val, right):
        self.val = val
        self.left = left
        self.right = right


# Creates inverted index based on the corpus
def create_index():
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


# Utility method to perform binary search for implementing next_pos
def binarysearch_high(term, low, high, current):
    while high - low > 1:
        mid = int((low + high) / 2)
        if posting_list[term][mid] <= current:
            low = mid
        else:
            high = mid
    return high


# Returns the next occurrence of a term given current position
def next_pos(term, current):
    # Query term not present in the inverted index
    if term not in posting_list.keys():
        return POSITIVE_INFINITY
    cache_next_pos[term] = -1
    length_posting = len(posting_list[term]) - 1
    if len(posting_list[term]) == 0 or posting_list[term][length_posting] <= current:
        return POSITIVE_INFINITY
    if posting_list[term][0] > current:
        cache_next_pos[term] = 0
        return posting_list[term][cache_next_pos[term]]
    if cache_next_pos[term] > 0 and posting_list[cache_next_pos[term] - 1] <= current:
        low = cache_next_pos[term] - 1
    else:
        low = 0
    jump = 1
    high = low + jump
    while high < length_posting and posting_list[term][high] <= current:
        low = high
        jump *= 2
        high = low + jump
    if high > length_posting:
        high = length_posting
    cache_next_pos[term] = binarysearch_high(term, low, high, current)
    return posting_list[term][cache_next_pos[term]]


# Returns the document id of the next document containing the term
def next_doc(term, current_doc):
    search_index = doc_first_last[current_doc][1]
    pos = next_pos(term, search_index)
    doc_num = docid(pos)
    return (doc_num)


# Utility method to perform binary search for implementing prev_pos
def binarysearch_low(term, low, high, current):
    while high - low > 1:
        mid = int((low + high) / 2)
        if posting_list[term][mid] >= current:
            high = mid
        else:
            low = mid
    return low


# Returns the previous occurrence of a term given current position
def prev_pos(term, current):
    # Query term not present in the inverted index
    if term not in posting_list.keys():
        return NEGATIVE_INFINITY
    cache_prev_pos[term] = len(posting_list[term])
    length_posting = len(posting_list[term]) - 1
    if len(posting_list[term]) == 0 or posting_list[term][0] >= current:
        return NEGATIVE_INFINITY
    if posting_list[term][length_posting] < current:
        cache_prev_pos[term] = length_posting
        return posting_list[term][cache_prev_pos[term]]
    if cache_prev_pos[term] < length_posting and posting_list[cache_prev_pos[term] + 1] >= current:
        high = cache_prev_pos[term] + 1
    else:
        high = length_posting
    jump = 1
    low = high - jump
    while low > 0 and posting_list[term][low] >= current:
        high = low
        jump *= 2
        low = high - jump
    if low < 0:
        low = 0
    cache_prev_pos[term] = binarysearch_low(term, low, high, current)
    return posting_list[term][cache_prev_pos[term]]


# Returns the document id of the previous document containing the term
def prev_doc(term, current_doc):
    if current_doc not in doc_first_last.keys():
        current_doc = POSITIVE_INFINITY
    search_index = doc_first_last[current_doc][0]
    pos = prev_pos(term, search_index)
    doc_num = docid(pos)
    return (doc_num)


# Creates a tree from reverse an expression in reverse polish notation
def create_tree(expression):
    list_exp = expression.split()
    return create_tree_helper(list_exp)


# Utility method to create the expression tree recursively
def create_tree_helper(expression):
    current = expression[0]
    expression.remove(current)
    if current not in [AND, OR]:
        return ExpressionTree(None, current, None)
    else:
        return ExpressionTree(create_tree_helper(expression), current, create_tree_helper(expression))


# Returns the end point of the first candidate solution after the current document
def doc_right(node, position):
    if node.left is None and node.right is None:
        return next_doc(node.val, position)
    elif node.val == AND:
        return max(doc_right(node.left, position), doc_right(node.right, position))
    elif node.val == OR:
        return min(doc_right(node.left, position), doc_right(node.right, position))


# Returns the starting point of the previous candidate solution before the current document
def doc_left(node, position):
    if node.left is None and node.right is None:
        return prev_doc(node.val, position)
    elif node.val == AND:
        return min(doc_left(node.left, position), doc_left(node.right, position))
    elif node.val == OR:
        return max(doc_left(node.left, position), doc_left(node.right, position))


# Finds the next valid document satisfying the query after the current position
def next_solution(query_tree, position):
    v = doc_right(query_tree, position)
    if v == POSITIVE_INFINITY:
        return POSITIVE_INFINITY
    u = doc_left(query_tree, v + 1)
    if u == v:
        return u
    else:
        return next_solution(query_tree, v)


# Generates a set of docuements satisfying the boolean query
def candidate_solutions(query_string):
    query_tree = create_tree(query_string)
    u = NEGATIVE_INFINITY
    while u < POSITIVE_INFINITY:
        u = next_solution(query_tree, u)
        if u < POSITIVE_INFINITY:
            valid_docs.append(u)
    return valid_docs


# Computes the term frequency for a given term
def get_tf(doc_id, term):
    for pair in inverted_index[term][:-1]:
        if pair[0] == doc_id:
            return float(1 + math.log(pair[1], 2))
    return 0.0

# Computes the inverse document frequency for a given term
def get_idf(term):
    return float(math.log(len(documents) / inverted_index[term][-1], 2))


# Generates the document vector
def compute_doc_vector():
    doc_vector = {}
    for doc_id in valid_docs:
        tmp_list = []
        for term in sorted(inverted_index.keys()):
            # Considering the total documents in the corpus to compute the tf and idf values
            tf = get_tf(doc_id, term)
            idf = get_idf(term)
            tmp_list.append(tf * idf)
        doc_vector[doc_id] = normalize(tmp_list)
    return doc_vector


# Utility method to normalize the document vector
def normalize(vector):
    length = math.sqrt(sum(map(lambda x: x * x, vector)))
    if length == 0.0:
        return []
    return list(map(lambda x: x / length, vector))


# Generates the query vector
def compute_query_vector():
    query_vector = []
    query_terms = query.translate(query.maketrans('', '', '_ANDOR')).split()
    for term in sorted(inverted_index.keys()):
        if term in query_terms:
            tf = float(1 + math.log(query_terms.count(term), 2))
            idf = get_idf(term)
            query_vector.append(tf * idf)
        else:
            query_vector.append(float(0))
    norm_query_vector = normalize(query_vector)
    # print(norm_query_vector)
    return normalize(norm_query_vector)


# Utility method to compute the dot product of the document and query vectors
def dot_product(doc_vector, query_vector):
    return sum(map(lambda x, y: x * y, doc_vector, query_vector))


# Utility method to return the next first document id satisfying the boolean query after the current document
def min_next_doc(doc_num):
    query_terms = query.translate(query.maketrans('', '', '_ANDOR')).split()
    tmp_docs = []
    for term in query_terms:
        tmp_docs.append(next_doc(term, doc_num))

    check_valid = sorted(tmp_docs)
    for d in sorted(tmp_docs):
        if d in valid_docs:
            return d
        elif d not in valid_docs:
            check_valid.remove(d)
            if len(check_valid) == 0:
                return POSITIVE_INFINITY


# Computes the tf-idf scores and retuens the top k results
def rank_cosine(k):
    norm_doc_vector = compute_doc_vector()
    norm_query_vector = compute_query_vector()
    d = min_next_doc(NEGATIVE_INFINITY)
    while d < POSITIVE_INFINITY:
        result[d] = dot_product(norm_doc_vector[d], norm_query_vector)
        d = min_next_doc(d)
    results = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return results


# Utility method to display the results of VSM
def display_results(k, results):
    if results is None or len(results) == 0:
        print("Query not found in the corpus.")
    else:
        print('DocID\tScore\n')
        for i in range(k):
            if i < len(results):
                print(str(results[i][0]) + '\t\t' + str(results[i][1]))
            else:
                print("\nThe total number of documents is " + str(
                    len(results)) + " which is less than the given value of k: " + str(k))
                break


# Utility method to separate the terms in a document
def separate_terms_in_documents(input_string):
    docs = input_string.split('\n\n')
    for i in range(len(docs)):
        docs[i] = docs[i].replace('\n', ' ')
        return docs

# Converts query terms to lower case
def normalize_query(query):
    tmp_query = []
    for x in query.split():
        if not (x == AND or x == OR):
            tmp_query.append(x.translate(str.maketrans('', '', string.punctuation)).lower())
        else:
            tmp_query.append(x)
    query = reduce(lambda x, y: x + ' ' + y, tmp_query)
    return query


def main():
    global documents
    global query
    global valid_docs

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
    create_index()

    # Generating a set of documents satisfying the given query
    valid_docs = candidate_solutions(query)


    # Displaying the top k solutions
    k = int(sys.argv[2])
    if len(valid_docs) > 0:
        results = rank_cosine(k)
    else:
        results = {}
    display_results(k, results)


if __name__ == '__main__':
    main()
