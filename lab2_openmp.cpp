#include <iostream>
#include <stdio.h>
#include <omp.h>

#define N 10000000
#define MAX_THREADS 8

using namespace std;

int arr[N];

 
void merge(int* a, int low, int mid, int high) {
    int i = low, j = mid + 1, size = 0;
    int* temp = new int[high - low + 1];
    for (; (i <= mid) && (j <= high); size++)
        if (a[i] < a[j])
            temp[size] = a[i++];
        else
            temp[size] = a[j++];
    while (i <= mid) temp[size++] = a[i++];
    for (i = 0; i < size; i++)
        a[low + i] = temp[i];
    delete[] temp;
}
 
void merge_sort(int* a, int low, int high) {
    if (low >= high)
        return;
    int mid = (low + high) / 2;
    merge_sort(a, low, mid);
    merge_sort(a, mid + 1, high);
    merge(a, low, mid, high);
}

void parallel_merge(int* a, int low, int mid, int high) {//[low, mid], [mid + 1, high]
    int seek_pos[MAX_THREADS + 1][3] = {}, n1 = mid - low + 1, n2 = high - mid;
    for (int i = 1; i < MAX_THREADS; i++) {
        int l1 = low, r1 = mid, pos1 = mid, pos2 = high;
        while (r1 - l1 > 0) {
            pos1 = (l1 + r1) / 2;
            int l2 = mid + 1, r2 = high + 1;
            while (r2 - l2 > 0) {
                pos2 = (l2 + r2) / 2;
                if (a[pos1] <= a[pos2]) r2 = pos2;
                else l2 = pos2 + 1;
            }
            pos2 = r2;
            if ((pos1 + pos2 - low - mid) * MAX_THREADS < (n1 + n2) * i) l1 = pos1 + 1;
            else r1 = pos1 - 1;
        }
        seek_pos[i][1] = pos1;
        seek_pos[i][2] = pos2;
        seek_pos[i][0] = seek_pos[i][1] + seek_pos[i][2] - low - mid - 1;
    }
    seek_pos[0][1] = low;
    seek_pos[0][2] = mid + 1;
    seek_pos[0][0] = seek_pos[0][1] + seek_pos[0][2] - low - mid - 1;
    seek_pos[MAX_THREADS][1] = mid + 1;
    seek_pos[MAX_THREADS][2] = high + 1;
    seek_pos[MAX_THREADS][0] = seek_pos[MAX_THREADS][1] + seek_pos[MAX_THREADS][2] - low - mid - 1;
 
    int* temp = new int[high - low + 1];
    #pragma omp parallel for num_threads(MAX_THREADS)
    for(int x = 0; x < MAX_THREADS; x++) {
        int i = seek_pos[x][1], j = seek_pos[x][2], k = seek_pos[x][0];
        while (i < seek_pos[x + 1][1] && j < seek_pos[x + 1][2])
            if (a[i] < a[j])
                temp[k++] = a[i++];
            else
                temp[k++] = a[j++];
        while (i < seek_pos[x + 1][1]) temp[k++] = a[i++];
        while (j < seek_pos[x + 1][2]) temp[k++] = a[j++];
    };
    #pragma omp parallel for num_threads(MAX_THREADS)
    for(int x = 0; x < MAX_THREADS; x++) {
        for (int i = seek_pos[x][0]; i < seek_pos[x + 1][0]; i++)
            a[low + i] = temp[i];
    };
    delete[] temp;
}

void parallel_merge_sort(int* a, int low, int high) {
    if (low >= high)
        return;
    int mid = (low + high) / 2;
    int dx = 1000;
    if (high - low > dx) {
        parallel_merge_sort(a, low, mid);
        parallel_merge_sort(a, mid + 1, high);
    }
    else {
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
    }
    merge(a, low, mid, high);
}

 
int main() {
    omp_set_num_threads(8);

    for (int i = 0; i < N; i++) {
        arr[i] = rand();
    }

    double start_time, end_time;

    start_time = omp_get_wtime();
    merge_sort(arr, 0, N - 1);
    end_time = omp_get_wtime();
    printf("Sequential merge sort time: %f seconds\n", end_time - start_time);

    start_time = omp_get_wtime();
    parallel_merge_sort(arr, 0, N - 1);
    end_time = omp_get_wtime();
    printf("Parallel merge sort time: %f seconds\n", end_time - start_time);

    return 0;
}