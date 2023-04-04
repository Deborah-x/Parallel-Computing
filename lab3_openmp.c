#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 100

void multiply(double A[][N], double B[][N], double C[][N]){
    int i, j, k;
    double sum;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}


void parallel_multiply(double A[][N], double B[][N], double C[][N]){
    clock_t start, end;
    start = omp_get_wtime();  // Start the timer
    // Perform the matrix multiplication using parallel computing
    #pragma omp parallel for
    for(int i = 0; i < 10000; i++){
        multiply(A, B, C);
    }
    end = omp_get_wtime();  // Stop the timer

    printf("Parallel time: %ld \n", end - start);
}

void no_parallel_multiply(double A[][N], double B[][N], double C[][N]){
    clock_t start, end;
    start = omp_get_wtime();  // Start the timer
    for(int i = 0; i < 10000; i++){
        multiply(A, B, C);
    }
    end = omp_get_wtime();  // Stop the timer

    printf("Serial time: %ld \n", end - start);
}

int main() {
    

    

    int i, j;
    double A[N][N], B[N][N], C[N][N];
    
    // Initialize the matrices A and B with random values
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (double) rand() / RAND_MAX;
            B[i][j] = (double) rand() / RAND_MAX;
        }
    }
    parallel_multiply(A, B, C);
    no_parallel_multiply(A, B, C);
    

    return 0;
}
