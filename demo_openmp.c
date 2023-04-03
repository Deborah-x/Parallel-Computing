#include <stdio.h>
#include <omp.h>
#include <time.h>

int main() {
    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Start the timer

    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }
    printf("Sum: %d\n", sum);

    end = clock();  // Stop the timer

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calculate the elapsed time

    printf("Elapsed time: %f seconds\n", cpu_time_used);

    return 0;
}
