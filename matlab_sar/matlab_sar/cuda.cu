/*GPU based IC-GN
* 2022-06-17 V1.0
* Coytright:Photomech Lab - Southeast University
* Author by:Y_ZY
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#define sub_R 15
#define M_PI    3.14159265358979323846

__global__ void generate_SAR4_kernel(float *SAR2 ,float  * ta, float * Az, float *Rg,float *SAR4,
									float vr, float H,  float fc, float c,float trs_min,
									float Fr, float L, float Nfast,
									int Na ) {
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	// int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= Na || j >= Na) {
		return;
	}
	float sum_re = 0.0f;
	float sum_im = 0.0f;
	for (int k = 0; k < 32; k++) {
		for (int m = 0; m < 32; m++) {
			float tav = (vr *ta[32 * k + m] - Az[i]);
			float Rt = sqrt(tav *tav + Rg[j] * Rg[j] + H*H);

			float tau = 2 * Rt / c;
			int nr = (round((tau - trs_min)*Fr*L));
			nr = nr < (int)L*(int)Nfast ? nr : (int)L*(int)Nfast;
			int y = m + 32 * k;
			int x = nr - 1;
			float rd_re = SAR2[(y *(int)(Nfast*L) + x) * 2];
			float rd_im = SAR2[(y *(int)(Nfast*L) + x) * 2 + 1];
			float a = 4 * M_PI*fc / c*Rt;
			sum_re += rd_re *cos(a) - rd_im *sin(a);
			sum_im += rd_re *sin(a) + rd_im *cos(a);

		}
	}
	SAR4[2 * (i * Na + j)] = sum_re;
	SAR4[2 * (i * Na + j) + 1] = sum_im;
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), file, line);
		exit(int(err));
	}
}

extern "C" void generate_vec_gpu(float *SAR2, float  * ta, float * Az, float *Rg, float *SAR4,
	float vr, float H, float fc, float c, float trs_min,
	float Fr, float L, float NFast,
	int Na , int fL ,int mL)
{
	//dim3 blockSize(128);
	//dim3 gridSize((imgH * imgW + blockSize.x - 1) / blockSize.x);

	int in_width = fL * 8;
	int in_height = mL;
	int in_size = in_width *in_height * 8;
	float * dev_SAR2, *dev_ta ,*dev_Az , *dev_Rg, *dev_SAR4;
	HANDLE_ERROR(cudaMalloc((void**)&dev_SAR2, in_size));
	HANDLE_ERROR(cudaMalloc((void**)&dev_ta, sizeof(float) * fL));

	HANDLE_ERROR(cudaMalloc((void**)&dev_Az, sizeof(float) * Na));
	HANDLE_ERROR(cudaMalloc((void**)&dev_Rg, sizeof(float) * Na));
	HANDLE_ERROR(cudaMalloc((void**)&dev_SAR4, Na *Na *8));

	HANDLE_ERROR(cudaMemcpy(dev_SAR2, SAR2, in_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_ta, ta, sizeof(float) * fL, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_Az, Az, sizeof(float) * Na, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_Rg, Rg, sizeof(float) * Na, cudaMemcpyHostToDevice));

	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	dim3 blockSize1(32, 32, 1);
	dim3 gridSize1((Na + blockSize1.x - 1) / blockSize1.x, (Na + blockSize1.y - 1) / blockSize1.y, (1 + blockSize1.z - 1) / blockSize1.z);
	//printf("gridSize =%d %d %d   blockSize =%d %d %d\n", gridSize1.x , gridSize1.y, gridSize1.z, blockSize1.x, blockSize1.y, blockSize1.z);
	generate_SAR4_kernel << <gridSize1, blockSize1 >> > (dev_SAR2, dev_ta, dev_Az, dev_Rg, dev_SAR4,
		vr, H, fc,  c,  trs_min, Fr,  L,  NFast, Na);
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	// std::cout << "generate_vec_gpu :" << msecTotal1 << std::endl;

	HANDLE_ERROR(cudaMemcpy(SAR4, dev_SAR4, Na *Na * 8, cudaMemcpyDeviceToHost));

	cudaFree(dev_SAR2);
	cudaFree(dev_ta);
	cudaFree(dev_Az);
	cudaFree(dev_Rg);
	cudaFree(dev_SAR4);
}



