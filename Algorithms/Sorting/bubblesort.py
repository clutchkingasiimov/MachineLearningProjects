#Bubble sort
def bubble_sort(array):
    print("Unsorted array: {}".format(array))
        
    for i in range(len(array)):
        for j in range(len(array)-1):
            if array[j] < array[j+1]:
                j += 1
            elif array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]

    print("Sorted array: {}".format(array))


array = [4, 7, 2, 1, 8, 3, 5, 19, 15, 17]
bubble_sort(array)
