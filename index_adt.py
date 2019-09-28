import sys
import string

PosInf = sys.maxsize
NegInf = -PosInf - 1

def polish_to_infix(query):

    stack = []
    AND='_AND'
    OR= '_OR'
    boolean_operators=[AND,OR]
    for term in reversed(query.split()):
        if term not in boolean_operators:
            stack.append(term)
        else:
            operand1 = stack.pop()
            operand2= stack.pop()
            stack.append('('+operand1+' '+term+' '+operand2+')')
    return stack[0]


def create_index(documents):
    inverted_index={}
    current_pos=1
    for line in documents:
        for word in line.split():
            if word not in inverted_index:
                inverted_index[word]=[1,[current_pos]]
            else:
                posting_list=inverted_index[word]
                posting_list[0]+=1
                posting_list[1] += [current_pos]
            current_pos+=1
    return inverted_index


def docid(position):
    if position == 0:
        return NegInf
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
        return PosInf


def binarySearch(term, low, high, current):
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
    if cache[term] > 0 and posting_list[cache[term]-1] < current:
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
    cache[term] = binarySearch(term, low, high, current)
    return posting_list[term][cache[term]]


def next_doc(term, current):
    pos = next_pos(term, current)
    doc_num = docid(pos)
    return (doc_num)

with open(sys.argv[1], 'r') as text:
    input_string = text.read()

input_string = input_string.translate(str.maketrans('', '', string.punctuation))
input_string = input_string.lower()
documents = input_string.split('\n\n')
for i in range(len(documents)):
    documents[i] = documents[i].replace('\n', ' ')

polish_query = sys.argv[2]
query = polish_to_infix(polish_query)

inv_index=create_index(documents)

posting_list = {}
for term in inv_index.keys():
    posting_list[term] = inv_index[term][1]
