AND = '_AND'
OR = '_OR'
query = "_OR _AND good dog _AND bad cat"
query = query.split(' ')
print(query)

class Tree:
    def __init__(self, left, val, right):
        self.val = val
        self.left = left
        self.right = right


def create_tree(expression):
    print(expression)
    current = expression[0]
    expression.remove(current)
    if current not in [AND, OR]:
        return Tree(None, current, None)
    else:
        return Tree(create_tree(expression), current, create_tree(expression))

def inorder(node):
    if node is not None:
        inorder(node.left)
        print(node.val)
        inorder(node.right)

inorder(create_tree(query))