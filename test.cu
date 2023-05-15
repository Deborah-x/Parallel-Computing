#include <stdio.h>

int main() 
{
    cudaDeviceProp mprop;
    cudaGetDeviceProperties(&mprop, 0);
    if(!mprop.deviceOverlap)
    {
        printf("Device not support overlaps, so stream is invalid!\n");
        
    }
    return 0;
}