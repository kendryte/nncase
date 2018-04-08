#ifndef _fft_h_
#define _fft_h_

#include <memory>
#include <array>
#include <cmath>
#include <complex>

#define M_PI       3.14159265358979323846   // pi

#define TRIG_PRECALC 0

#if TRIG_PRECALC
////// template class SinCosSeries
// common series to compile-time calculation of sine and cosine functions

template<unsigned M, unsigned N, unsigned B, unsigned A>
struct SinCosSeries {
	static double value() {
		return 1 - (A*M_PI / B)*(A*M_PI / B) / M / (M + 1)
			*SinCosSeries<M + 2, N, B, A>::value();
	}
};

template<unsigned N, unsigned B, unsigned A>
struct SinCosSeries<N, N, B, A> {
	static double value() { return 1.; }
};

////// template class Sin
// compile-time calculation of sin(A*M_PI/B) function

template<unsigned B, unsigned A, typename T = double>
struct Sin;

template<unsigned B, unsigned A>
struct Sin<B, A, float> {
	static float value() {
		return (A*M_PI / B)*SinCosSeries<2, 24, B, A>::value();
	}
};
template<unsigned B, unsigned A>
struct Sin<B, A, double> {
	static double value() {
		return (A*M_PI / B)*SinCosSeries<2, 34, B, A>::value();
	}
};

////// template class Cos
// compile-time calculation of cos(A*M_PI/B) function

template<unsigned B, unsigned A, typename T = double>
struct Cos;

template<unsigned B, unsigned A>
struct Cos<B, A, float> {
	static float value() {
		return SinCosSeries<1, 23, B, A>::value();
	}
};
template<unsigned B, unsigned A>
struct Cos<B, A, double> {
	static double value() {
		return SinCosSeries<1, 33, B, A>::value();
	}
};

#endif

template<unsigned Levels, typename T = double>
class fft
{
	enum { N = 1 << Levels };
public:
	fft()
	{
		for (int i = 0; i < N / 2; i++)
		{
			cosTable[i] = std::cos(T(2 * M_PI * i / N));
			sinTable[i] = std::sin(T(2 * M_PI * i / N));
		}
	}

	void forward(std::array<std::complex<T>, N>& data)
	{
		// Bit-reversed addressing permutation
		for (int i = 0; i < N; i++)
		{
			const auto j = (int)((uint32_t)Reverse(i) >> (32 - Levels));
			if (j > i)
				std::swap(data[i], data[j]);
		}

		// Cooley-Tukey decimation-in-time radix-2 FFT
		for (size_t size = 2; size <= N; size *= 2)
		{
			const auto halfSize = size / 2;
			const auto tableStep = N / size;

			for (size_t i = 0; i < N; i += size)
			{
				for (size_t j = i, k = 0; j < i + halfSize; j++, k += tableStep)
				{
					//auto temp = data[j + halfSize] * std::exp(std::complex<T>(0, -2 * M_PI * k / N));
					//data[j + halfSize] = data[j] - temp;
					//data[j] += temp;
					const auto h = j + halfSize;

					auto re = data[h].real();
					auto im = data[h].imag();

					auto tpre = +re * cosTable[k] + im * sinTable[k];
					auto tpim = -re * sinTable[k] + im * cosTable[k];

					auto rej = data[j].real();
					auto imj = data[j].imag();

					data[h] = std::complex<T>(rej - tpre, imj - tpim);
					data[j] = std::complex<T>(rej + tpre, imj + tpim);
				}
			}

			// Prevent overflow in 'size *= 2'
			if (size == N) break;
		}
	}
private:
	size_t reverseBits(size_t x)
	{
		size_t result = 0;
		for (int i = 0; i < N; i++, x >>= 1)
			result = (result << 1) | (x & 1U);
		return result;
	}

	int Reverse(int i)
	{
		i = ((i & 0x55555555) << 1) | (((uint32_t)i >> 1) & 0x55555555);
		i = ((i & 0x33333333) << 2) | (((uint32_t)i >> 2) & 0x33333333);
		i = ((i & 0x0f0f0f0f) << 4) | (((uint32_t)i >> 4) & 0x0f0f0f0f);
		i = (i << 24) | ((i & 0xff00) << 8) |
			((int)((uint32_t)i >> 8) & 0xff00) | (int)((uint32_t)i >> 24);
		return i;
	}

	std::array<T, N / 2> cosTable, sinTable;
};

#endif /* _fft_h_ */
