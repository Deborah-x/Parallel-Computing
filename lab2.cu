#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
int n = 10;


__global__ void quicksort(int *data, int left, int right)
{
    // int i, j, pivot, temp;
    printf("Enter\n");
    if (left < right)
    {
        // i = left;
        // j = right + 1;
        // pivot = data[left];

        // do {
        //     do {
        //         i++;
        //     } while (data[i] < pivot && i <= right);

        //     do {
        //         j--;
        //     } while (data[j] > pivot);

        //     if (i < j)
        //     {
        //         temp = data[i];
        //         data[i] = data[j];
        //         data[j] = temp;
        //     }
        // } while (i < j);

        // temp = data[left];
        // data[left] = data[j];
        // data[j] = temp;
        int i = left, j = right;
        int pivot = data[(left+right)/2];
        while(data[i] > pivot) i++;
        while(data[j] < pivot) j--;
        if(i < j) {
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        data[i] = pivot;
        
        // for (int i = 0; i < 10; i++)
        // {
        //     printf("%d ", data[i]);
        // }
        // printf("\n");
        
        quicksort<<<1, THREADS_PER_BLOCK>>>(data, left, j - 1);
        quicksort<<<1, THREADS_PER_BLOCK>>>(data, j + 1, right);
    }
}

void init(int *data) {
    for(int i = 0; i < n; i++) {
        data[i] = rand()%10;
    }

}

//串行快速排序
void sw(int *a, int *b){
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int *a, int sta, int end){
    int i = sta, j = end + 1;
    int x = a[sta];
    while (1)
    {
        while (a[++i] < x && i < end);
        while (a[--j] > x);
        if (i >= j)
            break;
        sw(&a[i], &a[j]);
    }
    a[sta] = a[j];
    a[j] = x;
    return j;
}

void quickSort(int *a, int sta, int end){
    if (sta < end){
        //printf("3\n");
        int mid = partition(a, sta, end);
        quickSort(a, sta, mid - 1);
        quickSort(a, mid + 1, end);
    }
}

int main()
{
    int *data;
    int *d_data;
    data = (int *)malloc(n*sizeof(int)); init(data);

    clock_t start_time, end_time;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);

    start_time = clock();
    quickSort(data, 0, n-1);
    end_time = clock();
    printf("CPU Time: %ld\n", end_time - start_time);
    for (int i = 0; i < n; i++)
    {
        printf("%d ", data[i]);
    }
    printf("\n");

    start_time = clock();
    quicksort<<<1, THREADS_PER_BLOCK>>>(d_data, 0, n - 1);
    end_time = clock();
    cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU Time: %ld\n", end_time - start_time);
    for (int i = 0; i < n; i++)
    {
        printf("%d ", data[i]);
    }
    printf("\n");
    cudaFree(d_data);

    return 0;
}