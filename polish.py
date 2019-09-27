
def polish_to_infix(query):
    stack= []
    AND='_AND'
    OR= '_OR'
    boolean_operators=[AND,OR]
    prev_op = None
    for term in reversed(query.split()):
        if term not in boolean_operators:
            stack.append(term)
        else:
            operand1 = stack.pop()
            operand2= stack.pop()
            stack.append('('+operand1+' '+term+' '+operand2+')')
    return stack[0]




polish_query="_OR _AND good dog _AND bad cat"
query=polish_to_infix(polish_query)
print(query)