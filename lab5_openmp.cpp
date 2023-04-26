#include "opencv2/opencv.hpp"
#include "omp.h"
#include "time.h"
#include <iostream>

#pragma comment(lib,"opencv_world300.lib")

using namespace cv;
using namespace std;


void normal(){
    clock_t start,end;
    Mat img = imread("test.jpg", IMREAD_GRAYSCALE);//读一张图(1920*1080)，转为灰度图//

    Mat out(img.rows,img.cols,CV_8U);//初始化输出图像//

    unsigned char* p=out.data;//像素指针//

    //取反//
    start = clock();
    for(int i = 0;i<img.rows*img.cols;i++){
        *p++ =0xff-img.data[i];
    }
    end = clock();

    cout<<"norm_time"<<(end-start)<<endl;


}

void test_omp(){

    clock_t start,end;
    Mat img = imread("test.jpg", IMREAD_GRAYSCALE);//读一张图，转为灰度图//

    Mat out(img.rows,img.cols,CV_8U);//初始化输出图像//

    unsigned char* p=out.data;//像素指针//
    //omp//
    int num = img.rows*img.cols;//openmp 限制循环格式//

    start =clock();
#pragma omp parallel for
    for (int i = 0;i<num;i++)
    {
        *p++ =0xff -img.data[i];
    }
    end  = clock();

    cout<<"omp_time"<<(end-start)<<endl;

}



int main(){

    normal();
    test_omp();

    return 0;

}