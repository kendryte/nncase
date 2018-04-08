#pragma once

#include <cmath>
#include <inttypes.h>
#include "fft.hpp"

template<typename T, size_t SampleRate = 16000, size_t WindowLength = 400, size_t CepstrumCount = 13, size_t FilterCount = 26, size_t FftSize = 512>
class mfcc
{
	enum { FftFreqNum = FftSize / 2 + 1 };

public:
	static_assert(FftSize == 512, "Only support 512 now.");
	static_assert(CepstrumCount <= FilterCount, "CepstrumCount must <= FilterCount");

	mfcc(T premph, float lowerFreqency = 0.0f, float upperFrequency = SampleRate / 2.f, float cepLifter = 22.f, float episilon = 1e-8f)
		:premph(premph), episilon(episilon), cepLifter(cepLifter)
	{
		build_filterbanks(lowerFreqency, upperFrequency);
	}

	void transform(const int16_t* signal, std::array<T, CepstrumCount>& features)
	{
		pre_emphasis(signal);
		fbank();

		auto energy = calc_energy();
		for (auto& feat : feats)
			feat = std::log(feat + episilon);
		dct_feats();
		lifter_feats();
		feats[0] = std::log(energy + episilon);

		for (size_t i = 0; i < CepstrumCount; i++)
			features[i] = feats[i];
	}

private:
	void pre_emphasis(const int16_t* signal)
	{
		for (size_t i = 1; i < WindowLength; i++)
			fftData[i] = signal[i] - signal[i - 1] * premph;
		fftData[0] = signal[0];

		for (size_t i = WindowLength; i < FftSize; i++)
			fftData[i] = {};
	}

	void fbank()
	{
		hamming();
		do_fft();
		filter_powerspec();
	}

	T calc_energy()
	{
		T energy = 0;
		for (size_t i = 0; i < FftFreqNum; i++)
			energy += powerSpec[i];
		return energy;
	}

	void hamming()
	{
		for (size_t i = 0; i < WindowLength; i++)
		{
			fftData[i] *= 0.54f - 0.46f * std::cos(T(2.0 * M_PI * i / (WindowLength - 1.0)));
		}
	}

	void do_fft()
	{
		fft_.forward(fftData);
		// Gen Power Spectrum
		for (size_t i = 0; i < FftFreqNum; i++)
		{
			auto mag = std::abs(fftData[i]);
			powerSpec[i] = mag * mag / FftSize;
		}
	}

	void build_filterbanks(float lowerFreqency, float upperFrequency)
	{
		const auto lowMel = mel(lowerFreqency);
		const auto highMel = mel(upperFrequency);

		std::array<float, FilterCount + 2> bins;
		{
			const auto step = (highMel - lowMel) / (bins.size() - 1);
			for (size_t i = 0; i < bins.size(); i++)
			{
				auto melPoint = lowMel + step * i;
				bins[i] = std::floor(melinv(melPoint) * (FftSize + 1) / SampleRate);;
			}
		}

		for (size_t i = 0; i < FilterCount; i++)
		{
			filterbanks[i] = {};
			auto begin = size_t(bins[i]);
			auto end = size_t(bins[i + 1]);
			for (size_t j = begin; j < end; j++)
				filterbanks[i][j] = (j - bins[i]) / (bins[i + 1] - bins[i]);

			begin = size_t(bins[i + 1]);
			end = size_t(bins[i + 2]);
			for (size_t j = begin; j < end; j++)
				filterbanks[i][j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1]);
		}
	}

	void filter_powerspec()
	{
		for (size_t i = 0; i < FilterCount; i++)
		{
			T sum = 0;
			for (size_t j = 0; j < FftFreqNum; j++)
				sum += powerSpec[j] * filterbanks[i][j];
			feats[i] = sum;
		}
	}

	float mel(float freq)
	{
		return 2595.0f * std::log10(1.0f + freq / 700.0f);
	}

	float melinv(float mel)
	{
		return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
	}

	void dct_feats()
	{
		std::array<T, FilterCount> result;
		const auto c = M_PI / (2.0f * FilterCount);
		const auto scale = std::sqrt(2.0f / FilterCount);

		for (int k = 0; k < FilterCount; k++)
		{
			T sum = 0;
			for (int n = 0; n < FilterCount; n++)
				sum += feats[n] * std::cos(T((2.0 * n + 1.0) * k * c));
			result[k] = scale * sum;
		}

		feats[0] = result[0] / std::sqrt(2.f);
		for (int i = 1; i < FilterCount; i++)
			feats[i] = result[i];
	}

	void lifter_feats()
	{
		for (size_t i = 0; i < CepstrumCount; i++)
			feats[i] *= 1 + cepLifter / 2 * std::sin(T(M_PI * i / cepLifter));
	}
private:
	T premph;
	T episilon;
	T cepLifter;

	std::array<std::complex<T>, FftSize> fftData;
	std::array<T, FftFreqNum> powerSpec;

	std::array<T, FilterCount> feats;

	// TODO: calc log
	fft<9, T> fft_;
	std::array<std::array<T, FftFreqNum>, FilterCount> filterbanks;
};