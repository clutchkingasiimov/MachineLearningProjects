#Binary Search Tree 
'''
Two key functions have been implemented: 

1. Search 
2. Insertion
'''

class Node:
    def __init__(self, value):
        self.value = value 
        self.right = None 
        self.left = None 

class BinarySearchTree:
    def __init__(self):
        self.root = None 

    #Insertion
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    #Insertion private function for recursion
    def _insert(self, value, current_node):
        if value < current_node.value:
            if current_node.left is None:
                current_node.left = Node(value)
            else:
                self._insert(value, current_node.left)
        if value > current_node.value:
            if current_node.right is None:
                current_node.right = Node(value)
            else:
                self._insert(value, current_node.right)
        
    #Searching node in the tree 
    def search(self, value):
        if self.root is value:
            print(f'Found {value} in the tree')
        else:
            self._search(value, self.root)

    def _search(self, value, current_node):
        if value == current_node.value:
            print(f'Found {value} in the tree')
        elif value < current_node.value and current_node.left is not None:
            self._search(value, current_node.left)
        elif value > current_node.value and current_node.right is not None:
            self._search(value, current_node.right)
        else:
            print(f'{value} does not exist in the tree')


    def print_tree(self):
        if self.root is not None:
            self._print_tree(self.root)

    def _print_tree(self, current_node):
        if current_node is not None:
            self._print_tree(current_node.left)
            print(str(current_node.value))
            self._print_tree(current_node.right)

numbers = [9, 8, 2, 3, 6, 7, 4, 11, 23]
tree = BinarySearchTree()
for num in numbers:
    tree.insert(num)

tree.print_tree()
tree.search(11)


