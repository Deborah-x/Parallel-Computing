//main.cu
#include "cuda_runtime.h"
 
#include <windows.h>   
#include <iostream>
 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
using namespace cv;
using namespace std;
 
void resizeImage(const Mat &_src, Mat &_dst, const Size &s )
{
	_dst = Mat::zeros(s, CV_8UC3);
	double fRows = s.height / (float)_src.rows;
	double fCols = s.width / (float)_src.cols;
	int pX = 0;
	int pY = 0;
	for (int i = 0; i != _dst.rows; ++i){
		for (int j = 0; j != _dst.cols; ++j){
			pX = cvRound(i/(double)fRows);
			pY = cvRound(j/(double)fCols);
			if (pX < _src.rows && pX >= 0 && pY < _src.cols && pY >= 0){
				_dst.at<Vec3b>(i, j)[0] = _src.at<Vec3b>(pX, pY)[0];
				_dst.at<Vec3b>(i, j)[1] = _src.at<Vec3b>(pX, pY)[1];
				_dst.at<Vec3b>(i, j)[2] = _src.at<Vec3b>(pX, pY)[2];
			}
		}
	}
}
 
bool initCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0){
		fprintf(stderr, "There is no device.\n");
		return false;
	}
 
	int i;
	for (i = 0; i < count; i++){
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess){
			if (prop.major >= 1){
				break;
			}
		}
	}
 
	if (i == count){
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
 
	cudaSetDevice(i);
	return true;
}
 
 
__global__ void kernel(uchar* _src_dev, uchar * _dst_dev, int _src_step, int _dst_step ,
	int _src_rows, int _src_cols, int _dst_rows, int _dst_cols)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
 
	double fRows = _dst_rows / (float)_src_rows;
	double fCols = _dst_cols / (float)_src_cols;
 
	int pX = 0;
	int pY = 0;
 
	pX = (int)(i / fRows);
	pY = (int)(j / fCols);
	if (pX < _src_rows && pX >= 0 && pY < _src_cols && pY >= 0){
		*(_dst_dev + i*_dst_step + 3 * j + 0) = *(_src_dev + pX*_src_step + 3 * pY);
		*(_dst_dev + i*_dst_step + 3 * j + 1) = *(_src_dev + pX*_src_step + 3 * pY + 1);
		*(_dst_dev + i*_dst_step + 3 * j + 2) = *(_src_dev + pX*_src_step + 3 * pY + 2);
	
	}
 
}
 
 
void resizeImageGpu(const Mat &_src, Mat &_dst, const Size &s)
{
	_dst = Mat(s, CV_8UC3);
	uchar *src_data = _src.data;
	int width = _src.cols;
	int height = _src.rows;
	uchar *src_dev , *dst_dev;
 
	cudaMalloc((void**)&src_dev, 3 * width*height * sizeof(uchar) );
	cudaMalloc((void**)&dst_dev, 3 * s.width * s.height * sizeof(uchar));
	cudaMemcpy(src_dev, src_data, 3 * width*height * sizeof(uchar), cudaMemcpyHostToDevice);
 
	double fRows = s.height / (float)_src.rows;
	double fCols = s.width / (float)_src.cols;
	int src_step = _src.step;
	int dst_step = _dst.step;
 
	dim3 grid(s.height, s.width);
	kernel << < grid, 1 >> >(src_dev, dst_dev, src_step, dst_step, height, width, s.height, s.width);
 
	cudaMemcpy(_dst.data, dst_dev, 3 * s.width * s.height * sizeof(uchar), cudaMemcpyDeviceToHost);
}
 
 
int main()
{
	Mat src = cv::imread("./images/lena.jpg" , 1);
	Mat dst_cpu;
 
	double start = GetTickCount();
	resizeImage(src, dst_cpu, Size(src.cols * 50, src.rows * 50));
	double  end = GetTickCount();
	
	cout << "CPU Time: " << end - start << "\n";
 
	initCUDA();
 
	Mat dst_gpu;
 
	start = GetTickCount();
	resizeImageGpu(src, dst_gpu, Size(src.cols * 50, src.rows * 50));
	end = GetTickCount();
	cout << "GPU Time: " << end - start << "\n";
 
	cv::imshow("Demo", dst_cpu);
	waitKey(0);
 
	return 0;
}