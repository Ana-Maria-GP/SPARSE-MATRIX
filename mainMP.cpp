#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("Usage: %s <n> <d> <m> <s> <nt>\n", argv[0]);
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
        printf("Execution time (CPU): %f seconds\n", end_time - start_time);
    } else {
        printf("Mode not recognized. Use 0 for CPU.\n");
        return 1;
    }

    // Free memory
    free(Md.elements);
    free(v);
    free(result);

    return 0;
}

void generateSparseMatrix(SparseMatrix* Md, float density, int seed) {
    srand(seed);

    Md->n = (int)(100 * density);  // adjust the size as needed
    Md->nnz = (int)(Md->n * Md->n * density);

    Md->elements = (SparseElement*)malloc(Md->nnz * sizeof(SparseElement));

    for (int i = 0; i < Md->nnz; ++i) {
        Md->elements[i].row = rand() % Md->n;
        Md->elements[i].col = rand() % Md->n;
        Md->elements[i].value = ((float)rand() / RAND_MAX) * 10.0;  // random values between 0 and 10
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
