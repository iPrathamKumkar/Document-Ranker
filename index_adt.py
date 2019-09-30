# Importing required libraries
import sys
import string
import math

# Defining the start and end of the corpus
PosInf = sys.maxsize
NegInf = -PosInf - 1

# Storing boolean operators as constants
AND='_AND'
OR= '_OR'

# Stores the inverted index for the corpus
inverted_index = {}

# Stores the posting list for the terms
posting_list = {}

# Stores the starting and ending position for each document
doc_first_last = {}

# Stores the set of documents satisfying the positive query
valid_docs = []

# Stores the results of the VSM computations
result = {}


# Defining a class to create a binary tree for query processing
class Tree:
    def __init__(self, left, val, right):
        self.val = val
        self.left = left
        self.right = right


# Creating inverted index
def create_index(documents):
    current_doc = 1
    total_count = {}
    for doc in documents:
        appeared = []
        for word in doc.split():
            if word not in inverted_index:
                inverted_index[word] = [[current_doc, 1]]
                appeared.append(word)
                total_count[word] = 1
            elif word not in appeared:
                inverted_index[word].append([current_doc, 1])
                appeared.append(word)
                total_count[word] += 1
            else:
                inverted_index[word][-1][1] += 1
        current_doc += 1
    for word in total_count.keys():
        inverted_index[word].append(total_count[word])
    create_posting(documents)
    store_first_last_doc(documents)

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


def docid(position):
    if position == NegInf:
        return NegInf
    if position == PosInf:
        return PosInf
    doc_num = 1
    prev_length = 0
    for doc in documents:
        words = doc.split(' ')
        doc_length = len(words) + prev_length
        if position <= doc_length:
            return doc_num
        else:
            prev_length = doc_length
            doc_num += 1
    return None


def store_first_last_doc(documents):
    prev_length = 0
    for i in range(1, len(documents) + 1):
        words = documents[i-1].split()
        doc_length = prev_length + len(words)
        doc_first_last[i] = (prev_length + 1, doc_length)
        prev_length = doc_length
    doc_first_last[NegInf] = (NegInf, 0)
    doc_first_last[PosInf] = (doc_first_last[len(documents)][1] + 1, PosInf)


def binarysearch_high(term, low, high, current):
    while high - low > 1:
        mid = int((low + high) / 2)
        if posting_list[term][mid] <= current:
            low = mid
        else:
            high = mid
    return high


def next_pos(term, current):
    cache = {}
    cache[term] = -1
    length_posting = len(posting_list[term]) - 1
    if len(posting_list[term]) == 0 or posting_list[term][length_posting] <= current:
        return PosInf
    if posting_list[term][0] > current:
        cache[term] = 0
        return posting_list[term][cache[term]]
    if cache[term] > 0 and posting_list[cache[term]-1] <= current:
        low = cache[term] - 1
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
    cache[term] = binarysearch_high(term, low, high, current)
    return posting_list[term][cache[term]]


def next_doc(term, current_doc):
    search_index = doc_first_last[current_doc][1]
    pos = next_pos(term, search_index)
    doc_num = docid(pos)
    return (doc_num)


def binarysearch_low(term, low, high, current):
    while high - low > 1:
        mid = int((low + high) / 2)
        if posting_list[term][mid] >= current:
            high = mid
        else:
            low = mid
    return low


def prev_pos(term, current):
    cache = {}
    cache[term] = len(posting_list[term])
    length_posting = len(posting_list[term]) - 1
    if len(posting_list[term]) == 0 or posting_list[term][0] >= current:
        return NegInf
    if posting_list[term][length_posting] < current:
        cache[term] = length_posting
        return posting_list[term][cache[term]]
    if cache[term] < length_posting and posting_list[cache[term] + 1] >= current:
        high = cache[term] + 1
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
    cache[term] = binarysearch_low(term, low, high, current)
    return posting_list[term][cache[term]]


def prev_doc(term, current_doc):
    if current_doc not in doc_first_last.keys():
        current_doc = PosInf
    search_index = doc_first_last[current_doc][0]
    pos = prev_pos(term, search_index)
    doc_num = docid(pos)
    return (doc_num)


def create_tree(expression):
    list_exp = expression.split(' ')
    return create_tree_helper(list_exp)


def create_tree_helper(expression):
    current = expression[0]
    expression.remove(current)
    if current not in [AND, OR]:
        return Tree(None, current, None)
    else:
        return Tree(create_tree_helper(expression), current, create_tree_helper(expression))


def inorder(node):
    if node is not None:
        inorder(node.left)
        print(node.val)
        inorder(node.right)


def assert_data (actual, expected):
    if expected == actual:
        print (str(actual))
    else:
        print("Fail - got ", str(actual))


def doc_right(node, position):
    if node.left is None and node.right is None:
        return next_doc(node.val, position)
    elif node.val == AND:
        return max(doc_right(node.left, position), doc_right(node.right, position))
    elif node.val == OR:
        return min(doc_right(node.left, position), doc_right(node.right, position))


def doc_left(node, position):
    if node.left is None and node.right is None:
        return prev_doc(node.val, position)
    elif node.val == AND:
        return min(doc_left(node.left, position), doc_left(node.right, position))
    elif node.val == OR:
        return max(doc_left(node.left, position), doc_left(node.right, position))


def next_solution(query_tree, position):
    v = doc_right(query_tree, position)
    if v == PosInf:
        return PosInf
    u = doc_left(query_tree, v+1)
    if u == v:
        return u
    else:
        return next_solution(query_tree, v)


def candidate_solutions(query_string):
    query_tree = create_tree(query_string)
    u = NegInf
    while u < PosInf:
        u = next_solution(query_tree, u)
        if u < PosInf:
            valid_docs.append(u)


def get_tf(doc_id, term):
    for pair in inverted_index[term][:-1]:
        if pair[0] == doc_id:
            return float(1 + math.log(pair[1], 2))
    return 0.0


def get_idf(term):
    return float(math.log(len(valid_docs)/inverted_index[term][-1], 2))


def compute_doc_vector():
    doc_vector = {}
    for doc_id in valid_docs:
        tmp_list = []
        for term in sorted(inverted_index.keys()):
            tf = get_tf(doc_id, term)
            idf = get_idf(term)
            tmp_list.append(tf*idf)
        doc_vector[doc_id] = normalize(tmp_list)
    print(doc_vector)
    return doc_vector


def normalize(vector):
    length = math.sqrt(sum(map(lambda x: x * x, vector)))
    return list(map(lambda x : x / length, vector))


def compute_query_vector():
    query_vector = []
    query_terms = query.translate(query.maketrans('', '', '_ANDOR')).split()
    for term in sorted(inverted_index.keys()):
        if term in query_terms:
            tf = float(1 + math.log(query_terms.count(term), 2))
            idf = get_idf(term)
            query_vector.append(tf*idf)
        else:
            query_vector.append(float(0))
    norm_query_vector = normalize(query_vector)
    print(norm_query_vector)
    return normalize(norm_query_vector)


def dot_product(doc_vector, query_vector):
    return sum(map(lambda x, y: x * y, doc_vector, query_vector))


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
                return PosInf


def rank_cosine(k):
    norm_doc_vector = compute_doc_vector()
    norm_query_vector = compute_query_vector()
    d = min_next_doc(NegInf)
    while d < PosInf:
        result[d] = dot_product(norm_doc_vector[d], norm_query_vector)
        d = min_next_doc(d)
    results = sorted(result.items(), key=lambda x: x[1], reverse=True)
    print('DocID\tScore\n')
    for i in range(k):
        if i < len(results):
            print(str(results[i][0])+'\t\t'+str(results[i][1]))
        else:
            print("\nThe total number of documents is " + str(len(results)) + " which is less than the given value of k: " + str(k))
            break


########################################################################################################################

# Reading the corpus file specified in command line
with open(sys.argv[1], 'r') as text:
    input_string = text.read()

# Removing punctuations
input_string = input_string.translate(str.maketrans('', '', string.punctuation))

# Converting to lowercase
input_string = input_string.lower()

# Splitting the corpus into documents and separating the terms
documents = input_string.split('\n\n')
for i in range(len(documents)):
    documents[i] = documents[i].replace('\n', ' ')

# Reading the positive query from command line
query = sys.argv[3]

# Creating an inverted index
create_index(documents)

# Generating a set of documents satisfying the given query
candidate_solutions(query)

# Returning the top k solutions
rank_cosine(5)
