all:
	nvcc -03 -Xcompiler -fopenmp main.cu -o prog