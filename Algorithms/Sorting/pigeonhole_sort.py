#Pigeonhole sort
def pigeonhole(array):

	"""
	Runs pigeonhole sort algorithm

	1. The range is defined by the formula max-min+1
	Time complexity: O(n+range)
	"""

    minimum, maximum = min(array), max(array)
    phole_range = maximum-minimum+1
    phole = [0] * phole_range

    #Sort the numbers by putting them into pigeonholes
    for i in range(len(array)):
        phole[array[i]-minimum] = array[i]

    #Sort them from the pigeonholes
    sorted_arr = []
    for i in range(len(phole)):
        if phole[i] != 0:
            sorted_arr.append(phole[i])
            i += 1

    print("Sorted array: {}".format(sorted_arr))

array = [5, 3, 9, 11, 2, 8, 17, 6]
pigeonhole(array)