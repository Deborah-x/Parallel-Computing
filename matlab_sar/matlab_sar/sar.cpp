// #include "stdafx.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "fft_1d.h"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

//#include "stdafx.h"
#include "fft_1d.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
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
	 memcpy(pVec, inVec, vecLen*sizeof(Complex)); 	// ����Ȩ������	
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
	 int r = get_computation_layers(vecLen); 	// ���㵹��λ��	
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
	 } 	// ������ٸ���Ҷ�任	

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
	 } 	// ʱ������д��X1
	 
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
				 X2_rl[index] = (X_rl[index] - X_rl[index - distance]) / 2;           // ��Ҫ�ٳ���W_rl[n*W]		
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
	 } 	// λ�뵹��
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


extern "C" void generate_vec_gpu(float *SAR2, float  * ta, float * Az, float *Rg, float *SAR4,
	float vr, float H, float fc, float c, float trs_min,
	float Fr, float L, float NFast,
	int Na, int fL, int mL);

using namespace std;
#define pi  3.141592653
void linspace(float begin, float finish, float *p_data ,int number)
{
	int j;
	float interval = (finish - begin) / (number - 1);

	for (j = 0; j < number; j++)
		p_data[j] = begin + j* interval;
}

void sar_test(float c, float fc, float vr, float H, float D, float Rg0, float RgL) {
	c = 3e8;
	fc = 7.5e8;                             
	float lambda = c / fc;                         
	vr = 100;                              
	H = 0;                             
	D = 8;                              
	Rg0 = 20480.0f;                            
	RgL = 2048.0f;           

	float R0 = sqrt(Rg0 *Rg0 + H *H);
	
	float La = lambda*R0 / D;
	// printf("La =%f \n" , La);

	float Ta = La / vr;

	float	Tw = 5e-6;                             
	float	Br = 15e6;                             
	float	Kr = Br / Tw;
	float	Fr = 2 * Br;
	float	Rmin = sqrt((Rg0 - RgL / 2) *(Rg0 - RgL / 2) + H *H);
	
	float	Rmax = sqrt((Rg0 + RgL / 2) *(Rg0 + RgL / 2) + H *H + (La / 2) *(La / 2));
	float	Nfast = ceil(2 * (Rmax - Rmin) / c*Fr + Tw*Fr); 
	//printf("Nfast =%f Rmax =%f Rmin =%f  %f %f %f %f  %f \n", Nfast, Rmax, Rmin,c, Fr, Tw , (Rg0 - RgL / 2) *(Rg0 - RgL / 2), (La / 2) *(La / 2));
	int nextpow2 = ceil(log2(Nfast));
	Nfast = pow(2, nextpow2);

	//tr = linspace(2 * Rmin / c, 2 * Rmax / c + Tw, Nfast);
	float *tr = (float *)malloc((int)Nfast * 4);
	linspace(2 * Rmin / c, 2 * Rmax / c + Tw, tr, (int)Nfast);

	float tr_min = tr[0];
	float tr_max = tr[1];
	for (int i = 1; i < (int)Nfast; i++) {
	
		tr_min = tr_min < tr[i] ? tr_min : tr[i];
		tr_max = tr_max > tr[i] ? tr_max : tr[i];
	}

	Fr = 1.0f / ((2 * Rmax / c + Tw - 2 * Rmin / c) / (int)Nfast); 


	float Az0 = 10240; //%��λ���������ʲô�趨��RD�㷨�о�û���趨
	float AL = 2048;
	float Azmin = Az0 - AL / 2;
	float Azmax = Az0 + AL / 2;
	//% ��λά / ��ʱ��ά����
	float Ka = -2 * vr*vr / lambda / R0;                // % ��ʱ��ά��Ƶ��
	float	Ba = fabs(Ka*Ta);                        //% ��ʱ��ά����
	float	PRF = 1.2*Ba;                           //% �����ظ�Ƶ��
	float	Mslow = ceil((Azmax - Azmin + La) / vr*PRF);  //% ��ʱ��ά���� / ������
	Mslow =ceil(log2(Mslow));
	//Mslow = Mslow *Mslow;              //% ������ʱ��άFFT�ĵ���
	Mslow = pow(2, Mslow);
	float *ta = (float *)malloc((int)Mslow * 4);
	linspace((Azmin - La / 2) / vr, (Azmax + La / 2) / vr, ta, (int)Mslow);
	// for (int i = 0; i < 10; i++) {
	// 	printf(" ta[ %d ] = %f \n ", i, ta[i]);
	// }

	PRF = 1 / ((Azmax - Azmin + La) / vr / Mslow);    //% ����ʱ��άFFT��������������ظ�Ƶ��
		
	float	Dr = c / 2 / Br;                            //% ����ֱ���
	float	Da = D / 2;                              // % ��λ�ֱ���

		//%% Ŀ�����
	int	Ntarget = 1;                            //% Ŀ������
	float	Ptarget[3] = { Az0 - 10, Rg0 - 20, 1 };           // % Ŀ��λ��\ɢ����Ϣ

	float sigmak = Ptarget[2];

	// printf("dddddddddddddddddddddddddddd \n");
	float *Srnm = (float *)malloc(Mslow *Nfast *8);
	for (int i = 0; i < Mslow; i++) {
	
		for (int j = 0; j < Nfast; j++) {

		float Azk = ta[i] * vr - Ptarget[0];
		float Rk = sqrt(Azk *Azk + Ptarget[1] * Ptarget[1] + H *H);

		float tauk = 2 * Rk / c;
		
		float tk =	tr[j] - tauk;

		float phasek = pi*Kr*tk*tk - (4 * pi / lambda)*Rk;
		bool  bool_0= (0 < tk&tk < Tw);
		bool bool_1 =(fabs(Azk) < La / 2);

		bool_0 = bool_0 &bool_1;

		Srnm[(i *(int)Nfast +j)*2] =sigmak *cos(phasek)*bool_0;
		Srnm[(i *(int)Nfast + j) * 2 + 1] = sigmak *sin(phasek)*bool_0;

		}
	
	}


#if 0
	cv::Mat out_Srnm(Nfast, Nfast, CV_32FC1);
	for (int i = 0; i < Nfast; i++) {
		for (int j = 0; j < Nfast; j++) {
			out_Srnm.ptr<float>(i)[j] = sqrt(Srnm[2 * (i * (int)Nfast + j)] * Srnm[2 * (i * (int)Nfast + j)] + Srnm[2 * (i * (int)Nfast + j) + 1] * Srnm[2 * (i * (int)Nfast + j) + 1]);
		}
	}
	cv::imshow("out_satr", out_Srnm);
	cv::waitKey();
#endif
	// printf("sssssssssssssssssssssssssss\n");
	float *hrc = (float *)malloc((int)Nfast * 8);
	Complex *inVec = new Complex[(int)Nfast];
	for (int i = 0; i < Nfast; i++) {
		float	thr = tr[i] - 2 * Rmin / c; //%�������źŷ���ʱ�����ɵ�ʱ������
		float a =pi*Kr*thr*thr;
		hrc[2 * i] = cos(a) *(0 < thr&thr < Tw);
		hrc[2 * i +1] = sin(a) *(0 < thr&thr < Tw);
		inVec[i].rl = hrc[2 * i];
		inVec[i].im = hrc[2 * i + 1];

	}
	Complex *outVec = new Complex[(int)Nfast];
	Complex *outVec_2 = new Complex[(int)Nfast];
	Complex *fft_out = new Complex[(int)Nfast];
	CFft1 t;
	t.fft(inVec, (int)Nfast, outVec);
	// printf("kkkkkkkkkkkkkkkkkkkkkk\n");
	float  *SAR1 = (float *)malloc((int)Nfast *(int)Mslow*8);
	for (int i = 0; i < Mslow; i++) {
		for (int j = 0; j < Nfast; j++) {
			inVec[j].rl = Srnm[(i *(int)Nfast + j) * 2] ;
			inVec[j].im = Srnm[(i *(int)Nfast + j) * 2 + 1] ;
		}
		t.fft(inVec, (int)Nfast, outVec_2);
		for (int j = 0; j < Nfast; j++) {
			//(outVec_2[j].rl + i *outVec_2[j].im)*(outVec[j].rl - i*outVec[j].im);
			fft_out[j].rl = outVec[j].rl *outVec_2[j].rl + outVec_2[j].im*outVec[j].im;
			fft_out[j].im = outVec_2[j].im *outVec[j].rl - outVec_2[j].rl *outVec[j].im;
		}
		t.ifft(fft_out, (int)Nfast, outVec_2);

		for (int j = 0; j < Nfast; j++) {
			SAR1[(i *(int)Nfast + j) * 2] = outVec_2[j].rl  ;
			SAR1[(i *(int)Nfast + j) * 2 + 1] = outVec_2[j].im;
		}
	}
	// printf("dddddddddddddddddddddddd\n");

#if 0
	cv::Mat out_SAR1(Nfast, Nfast, CV_32FC1);
	for (int i = 0; i < Nfast; i++) {
		for (int j = 0; j < Nfast; j++) {
			out_SAR1.ptr<float>(i)[j] = sqrt(SAR1[2 * (i * (int)Nfast + j)] * SAR1[2 * (i * (int)Nfast + j)] + SAR1[2 * (i * (int)Nfast + j) + 1] * SAR1[2 * (i * (int)Nfast + j) + 1]);
		}
	}
	cv::imshow("out_satr", out_SAR1);
	cv::waitKey();
#endif
	/*L = 8;%��ֵ����
trs = linspace(min(tr),max(tr),L*Nfast);
SAR1f = fft(SAR1,Nfast,2);

SAR11f = [SAR1f(:,1:floor((Nfast+1)/2)),zeros(Mslow,(L-1)*Nfast),...
    SAR1f(:,floor((Nfast+1)/2)+1:end)];%��ֵ����߲�����
SAR2 = ifft(SAR11f,L*Nfast,2);*/

	int L = 8;
	float *trs = (float *)malloc((int)Nfast *L * 4);

	linspace(tr_min, tr_max, trs, L*Nfast);

	float trs_min = trs[0];
	for (int i = 1; i < (int)Nfast *L; i++) {
		trs_min = trs_min < trs[i] ? trs_min : trs[i];
	}

	// printf("2222222222222222222222222\n");
	float  *SAR2 = (float *)malloc((int)Nfast *8 *(int)Mslow * 8);
	Complex *SAR1f_tmp = new Complex[(int)Nfast *8];
	Complex *SAR1f_tmp_out = new Complex[(int)Nfast * 8];

	for (int i = 0; i < Mslow; i++) {
		for (int j = 0; j < Nfast; j++) {
			inVec[j].rl = SAR1[(i *(int)Nfast + j) * 2];
			inVec[j].im = SAR1[(i *(int)Nfast + j) * 2 + 1];
		}
		t.fft(inVec, (int)Nfast, outVec_2);
		for (int j = 0; j < Nfast/2; j++) {
			
			SAR1f_tmp[j].rl = outVec_2[j].rl ;
			SAR1f_tmp[j].im = outVec_2[j].im ;
		}

		for (int j = Nfast / 2; j < (L -1 ) *Nfast + Nfast / 2; j++) {

			SAR1f_tmp[j].rl = 0.0f;
			SAR1f_tmp[j].im = 0.0f;
		}

		for (int j = (L - 1) *Nfast + Nfast / 2; j < (L) *Nfast; j++) {

			SAR1f_tmp[j].rl = outVec_2[j - (L - 1) *(int)Nfast].rl;
			SAR1f_tmp[j].im = outVec_2[j - (L - 1) *(int)Nfast].im;
		}

		t.ifft(SAR1f_tmp, (int)(Nfast *L), SAR1f_tmp_out);

		for (int j = 0; j < Nfast*L; j++) {
			SAR2[(i *(int)(Nfast*L) + j) * 2] = SAR1f_tmp_out[j].rl;
			SAR2[(i *(int)(Nfast*L) + j) * 2 + 1] = SAR1f_tmp_out[j].im;
		}
	}

#if 0
	{
	
	cv::Mat out_SAR1(Nfast, (int)(Nfast*L), CV_32FC1);
	for (int i = 0; i < Nfast; i++) {
		for (int j = 0; j < (int)(Nfast*L); j++) {
			out_SAR1.ptr<float>(i)[j] = sqrt(SAR2[2 * (i * (int)(Nfast*L) + j)] * SAR2[2 * (i * (int)(Nfast*L) + j)] + SAR2[2 * (i * (int)(Nfast*L) + j) + 1] * SAR2[2 * (i * (int)(Nfast*L) + j) + 1]);
		}
	}
	cv::resize(out_SAR1, out_SAR1, cv::Size(out_SAR1.cols / 8, out_SAR1.rows / 8));
	cv::imshow("out_satr2", out_SAR1);
	cv::waitKey();
	}
#endif
	// printf("44444444444444444444444444\n");
	//%% �����ʷ�
	//% ̽�ⷶΧ��ɢ��
	float *Rg = (float *)malloc(1001 *4);
	float *Az = (float *)malloc(1001 * 4);
	for (int i = 0; i < 1001; i++) {
		Rg[i] = Rg0 - 50 + 0.1f *i;
		Az[i] = Az0 - 50 + 0.1f *i;
	}
	int Nr = 1001;
	int Na = 1001;
	float  *SAR4 = (float *)malloc((int)Nr * Na * 8);

	struct timeval  start ,end;
	gettimeofday( &start, NULL );

	
	// printf("55555555 trs_min =%f Nfast =%f (Nfast*L) =%f \n", trs_min, (Nfast*L), (Nfast));
		for (int i = 0; i < Na; i++) {
			for (int j = 0; j < Nr; j++) {		
				float sum_re = 0.0f;
				float sum_im = 0.0f;
				for (int k = 0; k < 32; k++) {
					for (int m = 0; m < 32; m++) {
						float tav = (vr *ta[32 * k + m] - Az[i]);
						float Rt = sqrt(tav *tav + Rg[j] * Rg[j] + H*H);

						float tau = 2 * Rt / c;
						int nr = std::min(round((tau - trs_min)*Fr*L) ,(float)L*(int)Nfast);

						int y = m + 32 * k;
						int x = nr -1;

						float rd_re = SAR2[(y *(int)(Nfast*L) + x) * 2];
						float rd_im = SAR2[(y *(int)(Nfast*L) + x) * 2 + 1];

						float a = 4 * pi*fc / c*Rt;
						//cos(a) + sin(a);
						//exp(1j * 4 * pi*fc / c*Rt);
						//rd(m) = SAR2(m + Mslow / 32 * (k - 1), nr(m));

						sum_re += rd_re *cos(a) - rd_im *sin(a);
						sum_im += rd_re *sin(a) + rd_im *cos(a);

					}
				}
				SAR4[2 * (i * Na + j)] = sum_re;
				SAR4[2 * (i * Na + j)+ 1] = sum_im;
			
			}
	
		}
gettimeofday( &end, NULL );
float timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
printf("CPU Time: %f s\n" ,timeuse /1000000.0f);

gettimeofday( &start, NULL );
	generate_vec_gpu(SAR2, ta, Az, Rg, SAR4,
					vr, H, fc, c, trs_min,
					Fr, L, Nfast,
					Na, Nfast, Nfast);
gettimeofday( &end, NULL );					
 timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
printf("GPU Time: %f s\n" ,timeuse /1000000.0f);

		// printf("777777777777777 trs_min =%f \n", trs_min);
		//cv::Mat out(Na, Nr, CV_32FC1);
		//for (int i = 0; i < Na; i++) {
		//	for (int j = 0; j < Nr; j++) {
		//		out.ptr<float>(i)[j] =(sqrt(SAR4[2 * (i * Na + j)] * SAR4[2 * (i * Na + j)] + SAR4[2 * (i * Na + j) + 1] * SAR4[2 * (i * Na + j) + 1]))/255.0f;
		//	}
		//}
		//float Amin = *min_element(out.begin<float>(), out.end<float>());
		//float Amax = *max_element(out.begin<float>(), out.end<float>());
		//cv::Mat A_scaled = (out - Amin) / (Amax - Amin);
		//out.convertTo(A_scaled, CV_8UC1);
		//cv::applyColorMap(A_scaled, A_scaled, cv::COLORMAP_JET);
		//cv::cvtColor(A_scaled, A_scaled, CV_BGR2RGB);
		//("out_satr", out);
		//cv::waitKey();

}
	

int main()
{
	float c = 3e8;
	float fc = 7.5e8;                               //% �ź���Ƶ
	float lambda = c / fc;                         // % �ز�����
		//% ƽ̨����
	float vr = 100;                               //% SAR����ƽ̨�ٶ�
	float	H = 0;                               //% ƽ̨�߶�
		//%���߲���
	float	D = 8;                                  //% ��λ�����߳���

		//%% ��ʱ�����
		//% ������Χ
		float Rg0 = 20480;                            // % ���ĵؾ�
		float RgL = 2048;                            // % ������
	
	sar_test( c,  fc,  vr,  H,  D,  Rg0,  RgL);
	return 0;
}

