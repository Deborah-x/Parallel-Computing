#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
typedef long long ll;
typedef double lf;
struct Rader : vector<int>
{
	Rader(int n) : vector<int>(1 << int(ceil(log2(n))))
	{
		for (int i = at(0) = 0; i < size(); ++i)
			if (at(i) = at(i >> 1) >> 1, i & 1)
				at(i) += size() >> 1;
	}
};
struct FFT : Rader
{
	vector<complex<lf>> w;
	FFT(int n) : Rader(n), w(size(), polar(1.0, 2 * M_PI / size()))
	{
		w[0] = 1;
		for (int i = 1; i < size(); ++i)
			w[i] *= w[i - 1];
	}
	vector<complex<lf>> pfft(const vector<complex<lf>> &a) const
	{
		vector<complex<lf>> x(size());
#pragma omp parallel for
		for (int i = 0; i < a.size(); ++i)
			x[at(i)] = a[i];
		for (int i = 1; i < size(); i <<= 1)
#pragma omp parallel for
			for (int j = 0; j < i; ++j)
				for (int k = j; k < size(); k += i << 1)
				{
					complex<lf> t = w[size() / (i << 1) * j] * x[k + i];
					x[k + i] = x[k] - t, x[k] += t;
				}
		return x;
	}
	vector<complex<lf>> fft(const vector<complex<lf>> &a) const
	{
		vector<complex<lf>> x(size());
		for (int i = 0; i < a.size(); ++i)
			x[at(i)] = a[i];
		for (int i = 1; i < size(); i <<= 1)
			for (int j = 0; j < i; ++j)
				for (int k = j; k < size(); k += i << 1)
				{
					complex<lf> t = w[size() / (i << 1) * j] * x[k + i];
					x[k + i] = x[k] - t, x[k] += t;
				}
		return x;
	}
	vector<ll> ask(const vector<ll> &a, const vector<ll> &b) const
	{
		vector<complex<lf>> xa(a.begin(), a.end()), xb(b.begin(), b.end());
		xa = fft(xa), xb = fft(xb);
		for (int i = 0; i < size(); ++i)
			xa[i] *= xb[i];
		vector<ll> ans(size());
		xa = fft(xa), ans[0] = xa[0].real() / size() + 0.5;
		for (int i = 1; i < size(); ++i)
			ans[i] = xa[size() - i].real() / size() + 0.5;
		return ans;
	}
};
int main(int argc, char **argv)
{
	if (argc < 3)
		return cerr << "Error: No Enough parameters (" << argv[0] << " <num-of-threads> <power-of-transform-length>).\n", 0;
	omp_set_num_threads(atoi(argv[1]));
	FFT fft(1 << atoi(argv[2]));
	vector<complex<lf>> a(fft.begin(), fft.end());
	double t0 = omp_get_wtime();
	vector<complex<lf>> b = fft.fft(a);
	double t1 = omp_get_wtime();
	cout << "Sequential FFT time: " << t1 - t0 << " seconds\n";
	vector<complex<lf>> c = fft.pfft(a);
	double t2 = omp_get_wtime();
	cout << "Parallel FFT time:: " << t2 - t1 << " seconds\n";
	if (b != c)
		cerr << "Error: Parallel result are not equivalent to Serial result.\n";
}