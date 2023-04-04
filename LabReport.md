# Parallel Computing B-PB20000178 李笑
## Lab 1 - OpenMP及CUDA实验环境的搭建

以下两张截图是我个人电脑的配置：
![](images/config2.png)
![](images/config1.png)

切换到`Ubuntu20.04`进行后续实验，通过命令行查看配置：
![](images/CPU.png)


### 安装OpenMP
步骤：
1. 快捷键`Ctrl+Atl+T`打开终端
2. 在终端输入`sudo apt-get install libomp-dev`安装OpenMP
3. 在终端输入`sudo apt-get install gcc`安装GCC
4. 在终端输入`gcc --version`检查安装是否成功
![](images/GCC.png)
5. 在终端输入`echo |cpp -fopenmp -dM |grep -i open`检查OpenMP安装是否成功
![](images/OpenMP.png)


### 安装CUDA
配置前：
![](images/CUDA_before.png)
上图信息表明，我的电脑装有NVIDIA显卡，但是没有安装显卡驱动

步骤：
1. 手动安装显卡驱动。依次在终端输入，选择系统推荐版本驱动`nvidia-driver-525`

```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ ubuntu-drivers devices
$ sudo apt install nvidia-driver-525
```
![](images/nvidia-smi.png)
2. 关闭系统自带驱动`nouveau`。通过在终端输入指令`lsmod | grep nouveau`查看驱动启用情况。我输入后发现有输出，表明`nouveau`驱动正在工作。所以,接下来在终端输入`sudo gedit /etc/modprobe.d/blacklist.conf`，弹出了`blacklist.conf`文件，在文件末尾加上`blacklist nouveau`和`options nouveau modeset=0`两行并保存。
![](images/ban.png)
1. 重启
2. 进入NVIDIA官网CUDA下载页面`https://developer.nvidia.com/cuda-toolkit-archive`选择`CUDA Toolkit 11.2.0(December 2020)`，依次选择`Linux`→`x86_64`→`Ubuntu`→`20.04`→`runfile(local)`
3. 在终端输入`sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev`安装依赖库文件
4. 在终端输入`wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run`和`sudo sh cuda_11.2.0_460.27.04_linux.run`安装CUDA。接下来会弹出两个页面，在第一个页面输入`accept`、回车，在第二个页面按空格取消`Driver`勾选，然后点击`Install`、等待。
![](images/cuda.png)
5. 配置环境变量
```
$ export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
6. 在终端输入`source ~/.bashrc`使环境变量生效。
7. 查看CUDA安装信息
![](images/cuda_success.png)
8. CUDA测试。进入NVIDIA CUDA示例包，其位于`/home/xiaoli/NVIDIA_CUDA-11.2_Samples`，在该文件夹下打开终端，并输入`make`。然后进入`1_Utilities/deviceQuery`文件夹，并在终端执行`./deviceQuery`命令，输出结果`result=PASS`表示安装成功。
![](images/cuda_after.png)


## Lab 2 - 排序算法的并行及优化（验证）

## Lab 3 - 矩阵乘法的并行及优化（验证）

## Lab 4 - 快速傅里叶变换的并行实现（验证）

## Lab 5 - 常用图像处理算法的并行及优化（设计）

## Appendix
仅以此记录一下自己被困扰了一天的问题。以下是我写的第一个测试OpenMP的C语言代码

```C
#include <stdio.h>
#include <time.h>
#include <omp.h>

void sum(){
    int sum = 0;
    for(int i = 0; i < 100000000; i++){
        sum++;
    }
}

void parallel(){
    clock_t start, end;
    start = clock();
    # pragma omp parallel for
    for(int i = 0; i < 100; i++){
        sum();
    }
    end = clock();
    printf("Parallel time: %ld \n", end - start);
}

void no_parallel(){
    clock_t start, end;
    start = clock();
    for(int i = 0; i < 100; i++){
        sum();
    }
    end = clock();
    printf("Serial time: %ld \n", end - start);
}

int main() {
    parallel();
    no_parallel();
    return 0;
}
```

![](images/clock.png)

但是输出结果却令我大为震惊，因为开了并行竟然比不开更浪费时间，虽然我一开始以为可能是老师上课说的那种情况——并行的开销比并行的收益更大，但是当我把参数量调大之后发现这个现象仍然存在，于是我上网进行了搜索，终于发现原来是时间的测量方法使用错误。`clock()`记录的是CPU的滴答数，当并行多个进程同时计算，CPU滴答数成倍增加，所以我们得到的差值并不是真实的时间数，OpenMP提供的`omp_get_wtime()`才记录的是真实的运行时间，当我把所有时间测量函数从`clock()`修改为`omp_get_wtime()`后发现代码运行正常，结果如下
![](images/omp_get_wtime.png)
