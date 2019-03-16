using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Quantize
{
    /// <summary>ACIQ: ANALYTICALCLIPPING FORINTEGERQUAN-TIZATION OF NEURAL NETWORKS</summary>
    /// <seealso cref="https://openreview.net/pdf?id=B1x33sC9KQ"/>
    public static class Clipper
    {
        private const double _gausAlpha = 3.92403714;
        private static readonly double _gausConst = (0.5 * 0.35) * (1 + Math.Sqrt(Math.PI * Math.Log(4)));

        public static QuantizationRange GetClippedRange(ReadOnlySpan<float> data)
        {
            (var min, var max, var mean) = GetStatistics(data);
            var alpha = GetGaussianAlpha(min, max, data.Length);
            return GetClippedRange(min, max, mean, alpha);
        }

        private static (double min, double max, double mean) GetStatistics(ReadOnlySpan<float> data)
        {
            double min = double.MaxValue, max = double.MinValue;
            double sum = 0;
            for (int j = 0; j < data.Length; j++)
            {
                min = Math.Min(min, data[j]);
                max = Math.Max(max, data[j]);
                sum += data[j];
            }

            return (min, max, sum / data.Length);
        }

        private static double GetGaussianAlpha(double min, double max, int length)
        {
            double std = (max - min) * _gausConst / Math.Sqrt(2 * Math.Log(length));
            return _gausAlpha * std;
        }

        private static QuantizationRange GetClippedRange(double min, double max, double mean, double alpha)
        {
            double delta;

            var maxRange = max - min;
            if (alpha <= 0 || alpha >= maxRange / 2)
            {
                delta = maxRange;
            }
            else
            {
                delta = 2 * alpha;
                min = Math.Max(min, mean - delta / 2);
            }

            return new QuantizationRange { Min = min, Max = min + delta };
        }
    }
}
