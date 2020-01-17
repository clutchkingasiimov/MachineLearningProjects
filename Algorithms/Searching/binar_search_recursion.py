def binary_search(array, low, high, target):

	"""
	Performs binary search using recursion 
	on the partitioned arrays.
	"""
	if low > high:
		print("Search cancelled")

	middle = (low+high)//2 #Middle floor 

	if array[middle] == target:
		return middle 

	if array[middle] > target:
		return binary_search(array, low, middle-1, target)

	elif array[middle] < target:
		return binary_search(array, middle+1, high, target)

	else:
		print("Target not in array. Cancelling search")



numbers = [2, 4, 7, 13, 14, 35, 45, 78, 122, 545, 981, 1022]
position = binary_search(numbers, 0, len(numbers), 981)
print("Position of key is {}".format(position))