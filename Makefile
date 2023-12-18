all:
	nvcc main.cu -fopenmp -lcudart -lm -o prog
