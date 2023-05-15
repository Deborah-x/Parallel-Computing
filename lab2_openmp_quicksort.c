#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include "quickSort.h"
const int n = 10000000;
int main()
{
    omp_set_num_threads(8);
    double omp_time_sta, omp_time_end;
    double time_sta, time_end;
    int *data1, *data2;
    int num = 8;
    data1 = (int *)malloc(sizeof(int) * n);
    data2 = (int *)malloc(sizeof(int) * n);
    rands(data1, n);
	rands(data2, n);
    
    //并行快速排序
    omp_time_sta = omp_get_wtime();
    quickSort_parallel(data1, n, num);
	omp_time_end = omp_get_wtime();
    
    //串行快速排序
    time_sta = omp_get_wtime();
    quickSort(data2, 0, n - 1);
    time_end = omp_get_wtime();
    
    //输出运行时间
    printf("-------------------\n");
    printf("并行处理时间 : %lf s\n", omp_time_end - omp_time_sta);
    printf("串行处理时间 : %lf s\n", time_end - time_sta);
    printf("-------------------\n");
    // 输出排序后的数组
    // printf("-------------------\n");
    // printf("The final data1 : ");
    // print(data1, n);
    // printf("The final data2 : ");
    // print(data2, n);
    // printf("-------------------\n");
    // printf("\n\n");
    return 0;
}