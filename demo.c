#include <stdio.h>
#include <time.h>
#include <omp.h>

void sum(){
    int sum = 0;
    for(int i = 0; i < 100000000; i++){
        sum++;
    }
}

void parallel(){
    clock_t start, end;
    start = omp_get_wtime();
    # pragma omp parallel for
    for(int i = 0; i < 100; i++){
        sum();
    }
    end = omp_get_wtime();
    printf("Parallel time: %ld \n", end - start);
}

void no_parallel(){
    clock_t start, end;
    start = omp_get_wtime();
    for(int i = 0; i < 100; i++){
        sum();
    }
    end = omp_get_wtime();
    printf("Serial time: %ld \n", end - start);
}

int main() {
    parallel();
    no_parallel();
    return 0;
}
