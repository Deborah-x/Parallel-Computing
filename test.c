#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>


typedef struct Error {
	float max;
	float average;
}Error;

static void matMultCPU_serial(const float* , const float* , float* , int );
static void matMultCPU_parallel(const float* , const float* , float* , int );
void genMat(float* , int );
Error accuracyCheck(const float* , const float* , int );

int main(int argc, char** argv)
{
	/// test omp ///
	// #pragma omp parallel for
	// for (int i = 0; i < 10; i++) 
	// {
	// 	printf("Hello World %d from thread = %d\n", i, omp_get_thread_num());
	// }

	// Init matrix
	float* a, * b, * c, * d;
	int n = 1000;
	if (argc == 2) n = atoi(argv[1]);
	a = (float*)malloc(sizeof(float) * n * n);
	b = (float*)malloc(sizeof(float) * n * n);
	c = (float*)malloc(sizeof(float) * n * n);
	d = (float*)malloc(sizeof(float) * n * n);

	genMat(a, n);
	genMat(b, n);

	clock_t start, stop;
	start = omp_get_wtime();
	////// calculation code here ///////

	matMultCPU_serial(a, b, c, n);

	////// end code  ///////
	stop = omp_get_wtime();
	printf("CPU_Serial time: %3f ms\n", ((double)stop - start) / CLOCKS_PER_SEC * 1000.0);

	start = omp_get_wtime();
	////// calculation code here ///////

	matMultCPU_parallel(a, b, d, n);

	////// end code  ///////
	stop = omp_get_wtime();
	printf("CPU_Parallel time: %3f ms\n", ((double)stop - start) / CLOCKS_PER_SEC * 1000.0);

	return 0;
}

static void matMultCPU_serial(const float* a, const float* b, float* c, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double t = 0;
			for (int k = 0; k < n; k++)
			{
				t += (double)a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = t;
		}
	}
}

static void matMultCPU_parallel(const float* a, const float* b, float* c, int n)
{
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double t = 0;
			for (int k = 0; k < n; k++)
			{
				t += (double)a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = t;
		}
	}
}


void genMat(float* arr, int n)
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			arr[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX);
		}
	}
}


Error accuracyCheck(const float* a, const float* b, int n)
{
	Error err;
	err.max = 0;
	err.average = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (b[i * n + j] != 0)
			{
				//fabs求浮点数x的绝对值
				float delta = fabs((a[i * n + j] - b[i * n + j]) / b[i * n + j]);
				if (err.max < delta) err.max = delta;
				err.average += delta;
			}
		}
	}
	err.average = err.average / (n * n);
	return err;
}