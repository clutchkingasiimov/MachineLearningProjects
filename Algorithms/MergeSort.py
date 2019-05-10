def merge_sort(array):
    if len(array) > 1:
        split_point = int(len(array)/2)
        part_1 = array[:split_point]
        part_2 = array[split_point:]

        merge_sort(part_1)
        merge_sort(part_2)

        i = 0 
        j = 0 
        k = 0
        while i < len(part_1) and j < len(part_2):
            if part_1[i] < part_2[j]:
                array[k] = part_1[i]
                i +=1
            else:
                array[k] = part_2[j]
                j += 1
            k += 1
        
        while i < len(part_1):
            array[k] = part_1[i]
            i += 1
            k += 1

        while j < len(part_2):
            array[k] = part_2[j]
            j += 1
            k += 1

    return array 

numbers = [1, 8, 4, 5, 9, 2, 7, 6]
array = merge_sort(numbers)
print(array)
