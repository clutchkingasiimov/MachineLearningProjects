#Linear Search 
def linear_search(array, target):
    """
    Runs a linear search through the array and 
    finds the target key in the array 
    """
    for i in range(len(array)):
        if array[i] == target:
            print("Target found at index position {}".format(i+1))
            break


numbers = [4, 7, 6, 9, 3, 12, 55, 2]
linear_search(numbers, 9)
