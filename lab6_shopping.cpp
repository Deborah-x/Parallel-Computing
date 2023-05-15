#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include<omp.h>
#define NUM 28904 //总数据的数量
#define NUM1 5780 //测试数据的数量 28904*0.2 5780
#define NUM2 23124 //训练数据的数量 28904*0.8 23124
#define N 10 //特征数据的数量（维数）
#define KN 15//K的最大取值

typedef struct {
	double data;//距离
	char trainlabel;//用于链接训练标签

}Distance;

typedef struct {
	int data[N];
	int label;
}TestAndTrain;

TestAndTrain test[NUM1];//测试数据结构体数组
TestAndTrain train[NUM2];//训练数据结构体数组
TestAndTrain temp[NUM]; //临时存放数据结构体数组
Distance distance[NUM2];//存放距离结构体数组

void makerand(TestAndTrain a[],int n){ //函数功能：打乱存放标签后的结构体数组
	TestAndTrain t;
	int i=0,n1,n2;
	srand((unsigned int)time(NULL));
	for(i=0;i<n;i++){
		n1 = (rand() % n);//产生n以内的随机数  n是数组元素个数
		n2 = (rand() % n);
		if(n1 != n2){ //若两随机数不相等 则下标为这两随机数的数组进行交换
			t = a[n1];
			a[n1] = a[n2];
			a[n2] = t;
		}
	}
	
}

void tempdata(char filename[]){//临时存放数据  用于先存放150个数据后再打乱
	FILE* fp = NULL;
	fp = fopen(filename, "r");
	int i=0,j=0;
	for(i=0;i<NUM;i++){
		for(j=0;j<N;j++){
			fscanf(fp ,"%d ",&temp[i].data[j]);
			fgetc(fp);
		}
		fscanf(fp, "%d",&temp[i].label);
	}

	makerand(temp,NUM);//打乱所有数据

	// for(i=0;i<5;i++){
    //     for(j=0;j<10;j++){
    //         printf("%d ",temp[i].data[j]);
    //     }
    //     printf(" = %d\n", temp[i].label);
    // }
	fclose(fp);
	fp = NULL;
}

void loaddata() { //加载数据      分割：测试NUM1组   训练NUM2组
	int i, j, n = 0, m = 0;
	// #pragma omp parallel for schedule(dynamic)
	for (i = 0; i < NUM; i++) {
		if (i < NUM1) { //存入测试集
			for (j = 0; j < N; j++) {
				// printf("i=%d  j=%d\n",i,j);
				test[n].data[j] = temp[i].data[j]; //存入花的四个特征数据
			}
			test[n].label = temp[i].label;//存入花的标签
			n++;
		}
		else { //剩下的行数存入训练集
			for (j = 0; j < N; j++) {
				train[m].data[j] = temp[i].data[j];//存入花的四个特征数据
			}
			train[m].label = temp[i].label;//存入花的标签
			m++;
		}
	}
	// printf("test:\n"); //把分割后的数据都打印出来  便于观察是否已经打乱
	// for(i=0;i<NUM1;i++){
	// 	for(j=0;j<N;j++){
	// 		printf("%lf ",test[i].data[j]);
	// 	}
	// 	printf("%c\n",test[i].label);
	// }
	// printf("\n\ntrain:\n");
	// for(i=0;i<NUM2;i++){
	// 	for(j=0;j<N;j++){
	// 		printf("%lf ",train[i].data[j]);
	// 	}
	// 	printf("%c\n",train[i].label);
	// }
}

double computedistance(int n1,int n2) { //计算距离
	double sum = 0.0;
	int i; int tid;
	int temp[10];
	// #pragma omp parallel num_threads(10) private(tid)
	// {
	// 	tid = omp_get_thread_num();
	// 	temp[tid] = pow(test[n1].data[tid] - train[n2].data[tid], 2.0);
	// 	#pragma omp critical
	// 	sum += temp[tid];
	// 	// printf("%d\n",tid);
	// }
	for (i = 0; i < N; i++) {
		sum += pow(test[n1].data[i] - train[n2].data[i], 2.0);
		// printf("i=%d\n",i);
	}
	return sqrt(sum);//返回距离
}

int max(int a, int b, int c) { //找出频数最高的 测试数据就属于出现次数最高的
	if(a>b && a>c) return 1;
	if(b>a && b>c) return 2;
	if(c>a && c>b) return 3;
	return 0;
}

void countlabel(int* sum ,int k, int n) { //统计距离最邻近的k个标签出现的频数
	int i;
	int sum1 = 0, sum2 = 0, sum3 = 0;
	for (i = 0; i < k; i++) {
		switch (distance[i].trainlabel) { //用Distance结构体指针p来取K个距离最近的标签来进行判断
			case 1:sum1++; break;
			case 2:sum2++; break;
			case 3:sum3++; break;
		}
	}
	if (max(sum1, sum2, sum3) == test[n].label) { //检测距离最近的k个标签与原测试标签是否符合  并统计
		(*sum)++; //统计符合的数量
	}
}

int cmp(const void* a, const void* b) { //快速排序qsort函数的cmp函数(判断函数)
	Distance A = *(Distance*)a;
	Distance B = *(Distance*)b;
	return A.data > B.data ? 1 : -1;
}

int main()
{
	omp_set_num_threads(8);
	double start_time, end_time;
    char filename[20]={"support_data.txt"};
	tempdata(filename);//加载临时数据->打乱数据
	loaddata();//加载打乱后的数据并分割
	int i, j;
	int k=KN; //k值
	int sum = 0;//用于统计距离最近的k个标签与原测试标签符合的数量


	start_time = omp_get_wtime();

	#pragma omp parallel for
	for (i = 0; i < NUM1; i++) {
		#pragma omp parallel for
		for (j = 0; j < NUM2; j++) {
			// printf("i = %d    j = %d\n", i, j);
			distance[j].data = computedistance(i,j);//把计算好的距离依次存入distance结构体数组中
			distance[j].trainlabel = train[j].label; //以上距离存入的同时也把训练集标签一起存入distance结构体数组中
		}
		qsort(distance, NUM2, sizeof(distance[0]), cmp); //用qsort函数从小到大排序测试数据与每组训练数据的距离
		countlabel(&sum, k, i); //统计距离测试集标签最近的标签出现频数
	}
	end_time = omp_get_wtime();
	// printf("Sequential Time: %f\n", end_time-start_time);
	printf("Parallel Time: %f\n", end_time-start_time);
	// printf("K = %d     P = %.1lf%%\n", k,100.0*(sum)/NUM1);
	sum = 0;//每次统计完后都赋值0  便于下一个测试数据统计
	
	return 0;
}