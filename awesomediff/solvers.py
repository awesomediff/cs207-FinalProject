# import 
def gradientDescent(func, initial, rate=0.01, precision=0.00001, iteration = 2000):
	# df = func
	count = 0
	current = initial
	while (step>precision and count <iteration):
		last = current
		current = current - rate*func(last)
		step = abs(current-last)
		count+=1

	return current