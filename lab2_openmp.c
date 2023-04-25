#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
  
    // Create temp arrays
    int L[n1], R[n2];
  
    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];
  
    // Merge the temp arrays back into arr[l..r]
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
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
}
  
void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        
        // Sort the left and right halves in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            merge_sort(arr, l, m);
            
            #pragma omp section
            merge_sort(arr, m + 1, r);
        }
  
        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}

int main() {
    omp_set_num_threads(8);
    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = (double) rand() / RAND_MAX;
    }
    int n = sizeof(arr) / sizeof(arr[0]);
  
    // Sequential merge sort
    double start_time = omp_get_wtime();
    merge_sort(arr, 0, n - 1);
    double end_time = omp_get_wtime();
  
    printf("Sequential merge sort time: %f seconds\n", end_time - start_time);
  
    // Parallel merge sort
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(arr, 0, n - 1);
    }
    end_time = omp_get_wtime();
  
    printf("Parallel merge sort time: %f seconds\n", end_time - start_time);
        
    return 0;
}