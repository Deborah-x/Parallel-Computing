#include <iostream>
#include "/usr/include/opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>

using namespace cv;
using namespace std;

int main() {
    Mat src, dst_cpu;

    src = imread("./images/lena.jpg");

    if (src.empty())
	{
		cout << "读取错误" << endl;
		return -1;
	}

    printf("Right\n");
    return 0;
}