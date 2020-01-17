#Binary search 
def binary_search(array, target):

	"""
	Runs a binary search across the array 
	by splitting the array from the centerpoint 
	and searching through the subsets.

	If the value of the search key is less than the
	item in the middle of the interval, narrow the 
	interval to the lower half. 

	Otherwise narrow it to upper half. 
	Iterate until the value is found or the interval is empty

	Condition: Needs to take a sorted array
	"""

	#Set floor
	L = 0 
	R = len(array)

	while L <= R:
		m = (L+R)//2 #Set floor position to be the center of the array 

		if array[m] < target: #Check if target is greater than center
			L = m+1 #If it is greater, increment floor 1 step and check the greater side

		elif array[m] > target:
			R = m-1 #If smaller, decrement floor 1 step and check the smaller side

		else:
			return m  #If index == target, halt. 



numbers = [5,7,13,22,28,34,45]
output = binary_search(numbers, 13)
print("Position of the target: {}".format(output))




		






