import sys
import string

PosInf = sys.maxsize
NegInf = -PosInf - 1


class Tree:
    def __init__(self, left, val, right):
        self.val = val
        self.left = left
        self.right = right


AND='_AND'
OR= '_OR'

posting_list = {}
doc_first_last = {}


def create_posting(documents):
    current_pos = 1
    for line in documents:
        for word in line.split():
            if word not in posting_list:
                posting_list[word] = [current_pos]
            else:
                x = posting_list[word]
                x += [current_pos]
            current_pos += 1
    return posting_list


def docid(position):
    if position == NegInf:
        return None
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


def doc_f_l(documents):
    prev_length = 0
    for i in range(1, len(documents) + 1):
        words = documents[i-1].split()
        doc_length = prev_length + len(words)
        doc_first_last[i] = (prev_length + 1, doc_length)
        prev_length = doc_length


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
    search_index = doc_first_last[current_doc][1]
    pos = prev_pos(term, search_index)
    doc_num = docid(pos)
    return (doc_num)


def create_tree(expression):
    list_exp = expression.split(' ')
    return create_tree_helper(list_exp)


def create_tree_helper(expression):
    print(expression)
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
    elif node.val is AND:
        return max((doc_right(node.left), position), doc_right(node.right, position))
    elif node.val is OR:
        return min((doc_right(node.left), position), doc_right(node.right, position))


def doc_left(node, position):
    if node.left is None and node.right is None:
        return prev_doc(node.val, position)
    elif node.val is AND:
        return min((doc_left(node.left), position), doc_left(node.right, position))
    elif node.val is OR:
        return max((doc_left(node.left), position), doc_left(node.right, position))


with open(sys.argv[1], 'r') as text:
    input_string = text.read()

input_string = input_string.translate(str.maketrans('', '', string.punctuation))
input_string = input_string.lower()
documents = input_string.split('\n\n')
for i in range(len(documents)):
    documents[i] = documents[i].replace('\n', ' ')

query = sys.argv[2]
posting_list = create_posting(documents)
doc_f_l(documents)

