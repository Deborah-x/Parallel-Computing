#include "stdafx.h"
#include "fft_1d.h"
#include <stdio.h>
#include <string.h>
#include <math.h>


CFft1::CFft1() {} 
CFft1::~CFft1() {} 

bool CFft1::is_power_of_two(int IN num) 
{ 
	int temp = num;	int mod = 0;	
	int result = 0; 	
	if (num < 2)		
		return false;	
	if (num == 2)		
	   return true; 	
	while (temp > 1) { 
		result = temp / 2;		
		mod = temp % 2;		
		if (mod)			
			return false;	
		if (2 == result)			
			return true;		
		temp = result; 
	}	
	return false;
} 

int CFft1::get_computation_layers(int IN num)
{ 
	int nLayers = 0;	
	int len = num;	
	if (len == 2)	
		return 1;	
	while (true) { 
		len = len / 2;	
		nLayers++;	
		if (len == 2)			
			return nLayers + 1;	
		if (len < 1)			
			return -1;
	} 
} 


 bool CFft1::fft(Complex IN const inVec[], int IN const vecLen, Complex IN outVec[])
 {	
	 char msg[256] = "11111 "; 	
	 if ((vecLen <= 0) || (NULL == inVec) || (NULL == outVec))		
		 return false;	
	 if (!is_power_of_two(vecLen))		
		 return false; 	// create the weight array	
	 Complex         *pVec = new Complex[vecLen];	
	 Complex         *Weights = new Complex[vecLen];	
	 Complex         *X = new Complex[vecLen];	
	 int                   *pnInvBits = new int[vecLen]; 
	 memcpy(pVec, inVec, vecLen*sizeof(Complex)); 	// 计算权重序列	
	 float fixed_factor = (-2 * PI) / vecLen;
	 for (int i = 0; i < vecLen / 2; i++) 
	 {		
		 float angle = i * fixed_factor;	
	     Weights[i].rl = cos(angle);	
		 Weights[i].im = sin(angle);	
	 }	
	 for (int i = vecLen / 2; i < vecLen; i++) 
	 {	
		 Weights[i].rl = -(Weights[i - vecLen / 2].rl);	
		 Weights[i].im = -(Weights[i - vecLen / 2].im);
	 } 
	 int r = get_computation_layers(vecLen); 	// 计算倒序位码	
	 int index = 0;	
	 for (int i = 0; i < vecLen; i++) 
	 {		
		 index = 0;	
		 for (int m = r - 1; m >= 0; m--) 
		 {			
			 index += (1 && (i & (1 << m))) << (r - m - 1);		
		 }		
		 pnInvBits[i] = index;		
		 X[i].rl = pVec[pnInvBits[i]].rl;
		 X[i].im = pVec[pnInvBits[i]].im;	
	 } 	// 计算快速傅里叶变换	

	 for (int L = 1; L <= r; L++) 
	 {		
		 int distance = 1 << (L - 1);
		 int W = 1 << (r - L); 	
		 int B = vecLen >> L;	
		 int N = vecLen / B; 	
		 for (int b = 0; b < B; b++) 
		 {			
			 int mid = b*N;		
			 for (int n = 0; n < N / 2; n++) 
			 {				
				 int index = n + mid;	
				 int dist = index + distance;		
				 pVec[index].rl = X[index].rl + (Weights[n*W].rl * X[dist].rl - Weights[n*W].im * X[dist].im);  
				 // Fe + W*Fo				
				 pVec[index].im = X[index].im + Weights[n*W].im * X[dist].rl + Weights[n*W].rl * X[dist].im;		
			 }	
			 for (int n = N / 2; n < N; n++) 
			 {				
				 int index = n + mid;		
				 pVec[index].rl = X[index - distance].rl + Weights[n*W].rl * X[index].rl - Weights[n*W].im * X[index].im;        // Fe - W*Fo	
				 pVec[index].im = X[index - distance].im + Weights[n*W].im * X[index].rl + Weights[n*W].rl * X[index].im;			
			 }		
		 } 		
		 memcpy(X, pVec, vecLen*sizeof(Complex));	
	 } 	
	 memcpy(outVec, pVec, vecLen*sizeof(Complex));	
	 if (Weights)     
		 delete[] Weights;	
	 if (X)                
		 delete[] X;	
	 if (pnInvBits)    
		 delete[] pnInvBits;	
	 if (pVec)          
		 delete[] pVec;	
	 return true;
 } 

 bool CFft1::ifft(Complex IN const inVec[], int IN const len, Complex IN outVec[])
 {	
	 char msg[256] = "11111 "; 
	 if ((len <= 0) || (!inVec))	
		 return false;
	 if (false == is_power_of_two(len))
	 {		
		 return false;	
	 } 	
	 float         *W_rl = new float[len];
	 float         *W_im = new float[len];
	 float         *X_rl = new float[len];
	 float         *X_im = new float[len];	
	 float         *X2_rl = new float[len];	
	 float         *X2_im = new float[len]; 	
	 float fixed_factor = (-2 * PI) / len;
	 for (int i = 0; i < len / 2; i++) 
	 {		
		 float angle = i * fixed_factor;	
		 W_rl[i] = cos(angle);		
		 W_im[i] = sin(angle);	
	 }	
	 for (int i = len / 2; i < len; i++) 
	 {		
		 W_rl[i] = -(W_rl[i - len / 2]);	
		 W_im[i] = -(W_im[i - len / 2]);	
	 } 	// 时域数据写入X1
	 
	 for (int i = 0; i < len; i++) 
	 {		
		 X_rl[i] = inVec[i].rl;		
		 X_im[i] = inVec[i].im;	
	 }	
	 memset(X2_rl, 0, sizeof(float)*len);
	 memset(X2_im, 0, sizeof(float)*len); 
	 int r = get_computation_layers(len);	
	 if (-1 == r)		
		 return false;	
	 for (int L = r; L >= 1; L--) 
	 {		
		 int distance = 1 << (L - 1);	
		 int W = 1 << (r - L); 		
		 int B = len >> L;	
		 int N = len / B;		//sprintf(msg + 6, "B %d, N %d, W %d, distance %d, L %d", B, N, W, distance, L);	
								//OutputDebugStringA(msg); 		
		 for (int b = 0; b < B; b++) 
		 {			
			 for (int n = 0; n < N / 2; n++) 
			 {				
				 int index = n + b*N;		
				 X2_rl[index] = (X_rl[index] + X_rl[index + distance]) / 2;		
				 X2_im[index] = (X_im[index] + X_im[index + distance]) / 2;	
				 //sprintf(msg + 6, "%d, %d: %lf, %lf", n + 1, index, X2_rl[index], X2_im[index]);			
				 //OutputDebugStringA(msg);			
			 }			
			 
			 for (int n = N / 2; n < N; n++) 
			 {			
				 int index = n + b*N;			
				 X2_rl[index] = (X_rl[index] - X_rl[index - distance]) / 2;           // 需要再除以W_rl[n*W]		
				 X2_im[index] = (X_im[index] - X_im[index - distance]) / 2;		
				 float square = W_rl[n*W] * W_rl[n*W] + W_im[n*W] * W_im[n*W];          // c^2+d^2		
				 float part1 = X2_rl[index] * W_rl[n*W] + X2_im[index] * W_im[n*W];         // a*c+b*d			
				 float part2 = X2_im[index] * W_rl[n*W] - X2_rl[index] * W_im[n*W];          // b*c-a*d					
				 if (square > 0) 
				 {					
					 X2_rl[index] = part1 / square;			
					 X2_im[index] = part2 / square;			
				 }			
			 }		
		 }		
		 memcpy(X_rl, X2_rl, sizeof(float)*len);	
		 memcpy(X_im, X2_im, sizeof(float)*len);	
	 } 	// 位码倒序
	 int index = 0;
	 for (int i = 0; i < len; i++) 
	 {		
		 index = 0;		
		 for (int m = r - 1; m >= 0; m--) 
		 {			
			 index += (1 && (i & (1 << m))) << (r - m - 1);	
		 }		
		 outVec[i].rl = X_rl[index];	
		 outVec[i].im = X_im[index];	
		 //sprintf(msg + 6, "X_rl[i]: %lf, %lf,  index: %d", out_rl[i], out_im[i], index);	
		 //OutputDebugStringA(msg);	
	 } 	
	 if (W_rl)      
		 delete[] W_rl;	
	 if (W_im)   
		 delete[] W_im;
	 if (X_rl)     
		 delete[] X_rl;	
	 if (X_im)     
		 delete[] X_im;	
	 if (X2_rl)     
		 delete[] X2_rl;	
	 if (X2_im)    
		 delete[] X2_im; 
	 return true;
 }
