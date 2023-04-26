#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
			arr[i * n + j] = (float)rand() / RAND_MAX;
		}
	}
}



int main(int argc, char** argv)
{
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
	printf("Sequential matrix multiply time: %f seconds\n", (double)stop - start);

	start = omp_get_wtime();
	////// calculation code here ///////

	matMultCPU_parallel(a, b, d, n);

	////// end code  ///////
	stop = omp_get_wtime();
	printf("Parallel matrix multiply time: %f seconds\n", (double)stop - start);

	return 0;
}

