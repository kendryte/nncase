using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Data;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.Model;

namespace NnCase.Converter.K210.Converters.Stages.Quantize
{
    public struct QuantizationRange : IEquatable<QuantizationRange>
    {
        public static readonly QuantizationRange Default = new QuantizationRange { Min = -6, Max = 6 };

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

    public class ChannelwiseRange
    {
        public QuantizationRange[] Channels { get; set; }

        public QuantizationRange Global { get; set; }

        public ChannelwiseRange()
        {
        }

        public ChannelwiseRange(QuantizationRange range, int channels)
        {
            Global = range;
            Channels = Enumerable.Repeat(range, channels).ToArray();
        }

        public void EMA(double alpha, ChannelwiseRange newRange)
        {
            Global = Global.EMA(alpha, newRange.Global);
            for (int i = 0; i < Channels.Length; i++)
                Channels[i] = Channels[i].EMA(alpha, newRange.Channels[i]);
        }

        public void Union(ChannelwiseRange newRange)
        {
            Global = Global.Union(newRange.Global);
            for (int i = 0; i < Channels.Length; i++)
                Channels[i] = Channels[i].Union(newRange.Channels[i]);
        }
    }

    public class QuantizationContext
    {
        public GraphPlanContext PlanContext { get; set; }

        public IReadOnlyList<OutputConnector> Outputs { get; set; }

        public IReadOnlyList<Guid> AdditionalOutputs { get; set; }

        public Dictionary<OutputConnector, ChannelwiseRange> Distributions { get; } = new Dictionary<OutputConnector, ChannelwiseRange>();

        public Dictionary<Guid, ChannelwiseRange> AdditionalDistributions { get; } = new Dictionary<Guid, ChannelwiseRange>();

        public float Mean { get; set; }

        public float Std { get; set; }

        public bool ChannelwiseOutput { get; set; }
    }
}
