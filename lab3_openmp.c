#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1000

int main() {
    int i, j, k, num_threads;
    double sum;
    double A[N][N], B[N][N], C[N][N];
    printf("Enter");

    // Initialize the matrices A and B with random values
    for (i = 0; i < N; i++) {
        printf("i = %d\n", i);
        for (j = 0; j < N; j++) {
            A[i][j] = (double) rand() / RAND_MAX;
            B[i][j] = (double) rand() / RAND_MAX;
        }
    }

    // Set the number of threads to use
    // num_threads = 4;
    // omp_set_num_threads(num_threads);

    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Start the timer

    // Perform the matrix multiplication using parallel computing
    // #pragma omp parallel for private(j, k, sum)
    for (i = 0; i < N; i++) {
        printf("i = %d\n", i);
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    // Print the result matrix C
    // for (i = 0; i < N; i++) {
    //     for (j = 0; j < N; j++) {
    //         printf("%f ", C[i][j]);
    //     }
    //     printf("\n");
    // }

    end = clock();  // Stop the timer

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calculate the elapsed time

    printf("Elapsed time: %f seconds\n", cpu_time_used);

    return 0;
}
