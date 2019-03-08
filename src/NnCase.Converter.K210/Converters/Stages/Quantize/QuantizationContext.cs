using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.Model;

namespace NnCase.Converter.K210.Converters.Stages.Quantize
{
    public struct QuantizationRange : IEquatable<QuantizationRange>
    {
        public double Min;
        public double Max;

        public QuantizationRange EMA(double alpha, QuantizationRange range)
        {
            return new QuantizationRange { Min = alpha * range.Min + (1 - alpha) * Min, Max = alpha * range.Max + (1 - alpha) * Max };
        }

        public (double scale, double bias) GetScaleBias(int maxBits)
        {
            var scale = ((1 << maxBits) - 1) / (Max - Min);
            var bias = Math.Round(Min * scale);
            return (scale, bias);
        }

        public override string ToString()
        {
            return $"{Min}, {Max}";
        }

        public K210QuantizationParam GetQuantizationParam(int maxBits)
        {
            (var scale, var bias) = GetScaleBias(maxBits);
            return new K210QuantizationParam
            {
                Scale = (float)(1 / scale),
                Bias = (float)(bias / scale)
            };
        }

        public QuantizationRange Union(QuantizationRange range)
        {
            return new QuantizationRange { Min = Math.Min(Min, range.Min), Max = Math.Max(Max, range.Max) };
        }

        public override bool Equals(object obj)
        {
            return obj is QuantizationRange range && Equals(range);
        }

        public bool Equals(QuantizationRange other)
        {
            return Min == other.Min &&
                   Max == other.Max;
        }

        public static bool operator ==(QuantizationRange left, QuantizationRange right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(QuantizationRange left, QuantizationRange right)
        {
            return !(left == right);
        }
    }

    public class QuantizationContext
    {
        public GraphPlanContext PlanContext { get; set; }

        public IReadOnlyList<OutputConnector> Outputs { get; set; }

        public Dictionary<OutputConnector, QuantizationRange> Distributions { get; } = new Dictionary<OutputConnector, QuantizationRange>();
    }
}
