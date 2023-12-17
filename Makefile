CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3 -fopenmp
LDFLAGS = -lm -lcudart

all: prog

prog: main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o prog
