#include<iostream>
#include<opencv2/opencv.hpp>
#include<algorithm>
#include<omp.h>

using namespace cv;
using namespace std;

//构建检测核
Mat kernel1 = (cv::Mat_<float>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);//N
Mat kernel2 = (cv::Mat_<float>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);//NW
Mat kernel3 = (cv::Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);//W
Mat kernel4 = (cv::Mat_<float>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);//SW
Mat kernel5 = (cv::Mat_<float>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);//S
Mat kernel6 = (cv::Mat_<float>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);//SE
Mat kernel7 = (cv::Mat_<float>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);//E
Mat kernel8 = (cv::Mat_<float>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);//NE

int parallel_DIP() {
    Mat image, image_gray, image_bw, image_bw1, image_bw2, image_bw3;
	Mat image_bw4, image_bw5, image_bw6, image_bw7, image_bw8;

	image = imread("./images/lena.jpg");  //读取图像；
	if (image.empty())
	{
		cout << "读取错误" << endl;
		return -1;
	}

	//转换为灰度图像
	cvtColor(image, image_gray, COLOR_BGR2GRAY);
	// cv::imshow("image_gray", image_gray);
	
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            filter2D(image_gray, image_bw1, -1, kernel1); 
            convertScaleAbs(image_bw1, image_bw1);  
        }
        #pragma omp section
        {
            filter2D(image_gray, image_bw2, -1, kernel2); 
            convertScaleAbs(image_bw2, image_bw2); 
        }
        #pragma omp section 
        {
            filter2D(image_gray, image_bw3, -1, kernel3); 
            convertScaleAbs(image_bw3, image_bw3); 
        }
        #pragma omp section
        {
            filter2D(image_gray, image_bw4, -1, kernel4); 
            convertScaleAbs(image_bw4, image_bw4); 
        }
        #pragma omp section
        {
            filter2D(image_gray, image_bw5, -1, kernel5); 
            convertScaleAbs(image_bw5, image_bw5);  
        }
        #pragma omp section
        {
            filter2D(image_gray, image_bw6, -1, kernel6); 
            convertScaleAbs(image_bw6, image_bw6); 
        }
        #pragma omp section 
        {
            filter2D(image_gray, image_bw7, -1, kernel7); 
            convertScaleAbs(image_bw7, image_bw7); 
        }
        #pragma omp section
        {
            filter2D(image_gray, image_bw8, -1, kernel8); 
            convertScaleAbs(image_bw8, image_bw8); 
        }
    }
	
    image_bw = image_gray.clone();
    int i, j;
    #pragma omp parallel for shared(i, j) collapse(2)
	for (i = 0; i < image_gray.rows; i++)
	{
		for (j = 0; j < image_gray.cols; j++)
		{
			int arr[] = {image_bw1.at<uchar>(i, j), image_bw2.at<uchar>(i, j)
				, image_bw3.at<uchar>(i, j), image_bw4.at<uchar>(i, j), image_bw5.at<uchar>(i, j)
				, image_bw6.at<uchar>(i, j), image_bw7.at<uchar>(i, j), image_bw8.at<uchar>(i, j)};
			int max_num = *max_element(arr, arr + 8);
			image_bw.at<uchar>(i, j) = max_num;
		}
	}
	
	threshold(image_bw, image_bw, 220, 255, 0);
	// cv::imshow("image_bw", image_bw);
    return 1;
}

int DIP() {
    Mat image, image_gray, image_bw, image_bw1, image_bw2, image_bw3;
	Mat image_bw4, image_bw5, image_bw6, image_bw7, image_bw8;

	image = imread("./images/lena.jpg");  //读取图像；
	if (image.empty())
	{
		cout << "读取错误" << endl;
		return -1;
	}

	//转换为灰度图像
	cvtColor(image, image_gray, COLOR_BGR2GRAY);
	// cv::imshow("image_gray", image_gray);

    // 利用filter2D进行处理
	filter2D(image_gray, image_bw1, -1, kernel1);
	filter2D(image_gray, image_bw2, -1, kernel2);
	filter2D(image_gray, image_bw3, -1, kernel3);
	filter2D(image_gray, image_bw4, -1, kernel4);
	filter2D(image_gray, image_bw5, -1, kernel5);
	filter2D(image_gray, image_bw6, -1, kernel6);
	filter2D(image_gray, image_bw7, -1, kernel7);
	filter2D(image_gray, image_bw8, -1, kernel8);

	//绝对值
	convertScaleAbs(image_bw1, image_bw1);
	convertScaleAbs(image_bw2, image_bw2);
	convertScaleAbs(image_bw3, image_bw3);
	convertScaleAbs(image_bw4, image_bw4);
	convertScaleAbs(image_bw5, image_bw5);
	convertScaleAbs(image_bw6, image_bw6);
	convertScaleAbs(image_bw7, image_bw7);
	convertScaleAbs(image_bw8, image_bw8);

    image_bw = image_gray.clone();
    int i, j;
	for (i = 0; i < image_gray.rows; i++)
	{
		for (j = 0; j < image_gray.cols; j++)
		{
			int arr[] = {image_bw1.at<uchar>(i, j), image_bw2.at<uchar>(i, j)
				, image_bw3.at<uchar>(i, j), image_bw4.at<uchar>(i, j), image_bw5.at<uchar>(i, j)
				, image_bw6.at<uchar>(i, j), image_bw7.at<uchar>(i, j), image_bw8.at<uchar>(i, j)};
			int max_num = *max_element(arr, arr + 8);
			image_bw.at<uchar>(i, j) = max_num;
		}
	}
	
	threshold(image_bw, image_bw, 220, 255, 0);
    // cv::imshow("image_bw", image_bw);
    return 1;
}

int main() {
	omp_set_num_threads(8);
    double start_time, end_time;
    start_time = omp_get_wtime();
    if(DIP() < 0) {
        cout << "Something Wrong!" << endl;
    };
    end_time = omp_get_wtime();
    printf("Sequential DIP time: %f seconds\n", end_time - start_time);

    start_time = omp_get_wtime();
    if(parallel_DIP() <0) {
        cout << "Something Wrong!" << endl;
    }
    end_time = omp_get_wtime();
    printf("Parallel DIP time: %f seconds\n", end_time - start_time);

	// cv::waitKey(0);  //暂停，保持图像显示，等待按键结束
	return 0;
}