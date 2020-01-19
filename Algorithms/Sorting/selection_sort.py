#Selection sort 
def selection_sort(array):

    print("Unsorted array: {}".format(array))

    for i in range(len(array)-1):

        m = 0  #Set counter for the lowest value 
        for j in range(len(array)-1):
            if array[m] > array[j+1]:
                array[m], array[j+1] = array[j+1], array[m] #Swap
            m += 1 #Update counter position 

    print("Sorted array: {}".format(array))


array = [3, 18, 5, 1, 44, 32, 17, 42, 5]
selection_sort(array)


                
