#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>

#define PI 3.1415926535

float find(float n) {
    if(n <= 1) {
        n = 1;
        return n;
    }
    n = find(n/2);
    return n * 2;
}

void FFT(int n, int a[], double complex Y[]) {  // n is power of 2
    if(n == 1) {
        Y[0] = a[0];
        return;
    }
    int odd[n/2], even[n/2];
    for(int i = 0; i < n; i++) {
        if(i%2 == 0) even[i/2] = a[i];
        else odd[i/2] = a[i];
    }
    double complex S0[n], S1[n];
    FFT(n/2, even, S0);
    FFT(n/2, odd, S1);
    double complex z = (2*PI/n)*I;
    double complex w = cexp(z);
    double complex wk = 1;
    for(int k = 0; k < n/2; k++) {
        Y[k] = S0[k] + S1[k] * wk;
        Y[k+n/2] = S0[k] - S1[k] * wk;
        wk *= w;
    }
    return;
}

void IFFT(int n, double complex Y[], double complex y[]) {
    if(n == 1) {
        y[0] = Y[0];
        return;
    }
    double complex Y0[n/2], Y1[n/2];
    for(int i = 0; i < n; i++) {
        if(i%2 == 0) Y0[i/2] = Y[i];
        else Y1[i/2] = Y[i];
    }
    double complex y0[n], y1[n];
    IFFT(n/2, Y0, y0);
    IFFT(n/2, Y1, y1);
    double complex z = (2*PI/n)*I;
    double complex w = cexp(-z);
    double complex wk = 1;
    for(int k = 0; k < n/2; k++) {
        y[k] = y0[k] + y1[k] * wk;
        y[k+n/2] = y0[k] - y1[k] * wk;
        wk *= w;
    }
    
    return;
}

int main() {
    float num;
    scanf("%f", &num);
    int n = (int) find(num+1);
    
    int *a = (int*)calloc(2*n, sizeof(int));
    int *b = (int*)calloc(2*n, sizeof(int));
    int *c = (int*)calloc(2*n, sizeof(int));
    
    for(int i = 0; i < num+1; i++) {
        scanf("%d", &a[i]);
    }
    for(int i = 0; i < num+1; i++) {
        scanf("%d", &b[i]);
    }

    double complex *V = (double complex*)malloc(2*n * sizeof(double complex));
    double complex *U = (double complex*)malloc(2*n * sizeof(double complex));
    double complex *Y = (double complex*)malloc(2*n * sizeof(double complex));
    double complex *y = (double complex*)malloc(2*n * sizeof(double complex));
    FFT(2*n, a, V);
    FFT(2*n, b, U);
    for(int i = 0; i < 2*n; i++) {
        Y[i] = V[i] * U[i];
    }
    IFFT(2*n, Y, y);
    for(int i = 0; i < 2*n; i++) {
        y[i] /= 2*n;
    }
    for(int i = 0; i < 2*n; i++) {
        if(creal(y[i]) - floor(creal(y[i])) < 0.5) c[i] = (int)floor(creal(y[i]));
        else c[i] = (int)floor(creal(y[i])) + 1;
    }
    int tail = 2*n-1;
    while(c[tail] == 0) {
        tail--;
    }
    for(int i = 0; i <= tail; i++) {
        printf("%d\n", c[i]);
    }

    free(a);
    free(b);
    free(c);
    return 0;
}