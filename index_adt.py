import sys
import string


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

with open(sys.argv[1], 'r') as text:
    input_string = text.read()

input_string = input_string.translate(str.maketrans('', '', string.punctuation))

input_string = input_string.lower()
documents = input_string.split('\n\n')

for i in range(len(documents)):
    documents[i] = documents[i].replace('\n', ' ')
print(documents)

polish_query = sys.argv[2]
q = polish_to_infix(polish_query)
print(q)


def docid(position):
    if position == 0:
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


print(docid(0))

posting_list = {'a': [21], 'sir': [4, 6, 8 ,12, 28]}

def next_pos(term, position):
