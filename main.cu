#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

typedef struct {
    int row;
    int col;
    float value;
} SparseElement;

typedef struct {
    int n;
    int nnz;
    SparseElement* elements;
} SparseMatrix;

void generateSparseMatrix(SparseMatrix* Md, float density, int seed);
void generateVector(float* v, int n, int seed);

void sparseMatrixVectorMultiplyCPU(const SparseMatrix* Md, const float* v, float* result, int num_threads);
void sparseMatrixVectorMultiplyGPU(const SparseMatrix* Md, const float* v, float* result);

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("Uso: %s <n> <d> <m> <s> <nt>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    float density = atof(argv[2]);
    int mode = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int num_threads = atoi(argv[5]);

    SparseMatrix Md;
    generateSparseMatrix(&Md, density, seed);

    float* v = (float*)malloc(n * sizeof(float));
    generateVector(v, n, seed);

    float* result = (float*)malloc(n * sizeof(float));

    if (mode == 0) {
        omp_set_num_threads(num_threads);
        double start_time = omp_get_wtime();
        sparseMatrixVectorMultiplyCPU(&Md, v, result, num_threads);
        double end_time = omp_get_wtime();
        printf("Tiempo de ejecucion (CPU): %f segundos\n", end_time - start_time);
    } else if (mode == 1) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float* d_result;
        cudaMalloc((void**)&d_result, n * sizeof(float));
        cudaMemset(d_result, 0, n * sizeof(float));

        cudaEventRecord(start);
        sparseMatrixVectorMultiplyGPU(&Md, v, d_result);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Tiempo de ejecucion (GPU): %f segundos\n", milliseconds / 1000.0);

        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        printf("Modo no reconocido. Use 0 para CPU o 1 para GPU.\n");
        return 1;
    }

    // Liberar memoria
    free(Md.elements);
    free(v);
    free(result);

    return 0;
}

void generateSparseMatrix(SparseMatrix* Md, float density, int seed) {
    srand(seed);

    Md->n = (int)(100 * density);  // ajusta el tamaño según sea necesario
    Md->nnz = (int)(Md->n * Md->n * density);

    Md->elements = (SparseElement*)malloc(Md->nnz * sizeof(SparseElement));

    for (int i = 0; i < Md->nnz; ++i) {
        Md->elements[i].row = rand() % Md->n;
        Md->elements[i].col = rand() % Md->n;
        Md->elements[i].value = ((float)rand() / RAND_MAX) * 10.0;  // valores aleatorios entre 0 y 10
    }
}

void generateVector(float* v, int n, int seed) {
    srand(seed);

    for (int i = 0; i < n; ++i) {
        v[i] = ((float)rand() / RAND_MAX) * 10.0;
    }
}

void sparseMatrixVectorMultiplyCPU(const SparseMatrix* Md, const float* v, float* result, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < Md->n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < Md->nnz; ++j) {
            if (Md->elements[j].row == i) {
                sum += Md->elements[j].value * v[Md->elements[j].col];
            }
        }
        result[i] = sum;
    }
}

__global__ void sparseMatrixVectorMultiplyGPU(const SparseElement* elements, const float* v, float* result, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz) {
        atomicAdd(&result[elements[tid].row], elements[tid].value * v[elements[tid].col]);
    }
}
