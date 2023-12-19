all:
	nvcc smGPU.cu -lcudart -lm -o progG
	g++ -O3 -fopenmp -x c++ smCPU.cpp -o progC
gpu:
	nvcc smGPU.cu -lcudart -lm -o progG
cpu:
	g++ -O3 -fopenmp -x c++ smCPU.cpp -o progC
