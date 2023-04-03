#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define ARRAY_SIZE 100000

void merge(int *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
 
    // Create temporary arrays
    int *L = (int*) malloc(n1 * sizeof(int));
    int *R = (int*) malloc(n2 * sizeof(int));
 
    // Copy data to temporary arrays
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];
 
    // Merge the temporary arrays back into arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
 
    // Free temporary arrays
    free(L);
    free(R);
}
 
void mergeSort(int *arr, int l, int r) {
    if (l < r) {
        // Calculate mid-point
        int m = l + (r - l) / 2;
 
        // Parallelize left and right sub-arrays
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m);

            #pragma omp section
            mergeSort(arr, m + 1, r);
        }
 
        // Merge sorted sub-arrays
        merge(arr, l, m, r);
    }
}

int main() {
    int arr[ARRAY_SIZE];
    int i;

    srand(time(NULL));  // Seed the random number generator with the current time

    // Fill the array with random integers between 0 and 99
    for (i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % 100000;
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Start the timer

    // int arr[] = {38, 27, 43, 3, 9, 82, 10};
    // int n = sizeof(arr) / sizeof(arr[0]);

    // Sort array using parallel merge sort
    mergeSort(arr, 0, ARRAY_SIZE - 1);

    // Print sorted array
    // printf("Sorted array: ");
    // for (int i = 0; i < ARRAY_SIZE; i++) {
    //     printf("%d ", arr[i]);
    // }
    // printf("\n");

    end = clock();  // Stop the timer

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calculate the elapsed time

    printf("Elapsed time: %f seconds\n", cpu_time_used);

    return 0;
}
