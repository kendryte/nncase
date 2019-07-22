using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using NnCase.IR;

namespace NnCase.Evaluation
{
    public class Quantizer
    {
        private const float _floatEpsilon = 1.192092896e-07F;
        private const float _minQuantizeRange = 0.001f;

        private readonly Dictionary<OutputConnector, ValueRange<float>> _quantRnages = new Dictionary<OutputConnector, ValueRange<float>>();

        public ValueRange<float> GetRange(ReadOnlySpan<float> data)
        {
            float min = float.MaxValue;
            float max = float.MinValue;

            foreach (var v in data)
            {
                min = Math.Min(min, v);
                max = Math.Max(max, v);
            }

            return new ValueRange<float> { Min = min, Max = max };
        }

        public void Record(OutputConnector connector, ValueRange<float> range)
        {
            if (_quantRnages.TryGetValue(connector, out var oldRange))
                _quantRnages[connector] = Combine(oldRange, range);
            else
                _quantRnages[connector] = range;
        }

        public void Record(OutputConnector connector, ReadOnlySpan<float> data)
        {
            Record(connector, GetRange(data));
        }

        public ValueRange<float> Get(OutputConnector connector)
        {
            return _quantRnages[connector];
        }

        public QuantizationParam GetQuantizationParam(ValueRange<float> range, int bits)
        {
            if (range.Max < 0)
                range.Max = 0;
            if (range.Min > 0)
                range.Min = 0;

            var r = range.Max - range.Min;
            if (r < _minQuantizeRange)
                r = _minQuantizeRange;

            var scale = ((1L << bits) - 1) / r;
            var bias = MathF.Round(-range.Min * scale);
            Debug.Assert(bias >= 0);
            return new QuantizationParam { ZeroPoint = (int)bias, Scale = scale };
        }

        public FixedMul GetFixedMul(float value, int maxBits, int maxShift, bool isSigned)
        {
            Debug.Assert(!isSigned || value >= 0);

            var bits = isSigned ? maxBits - 1 : maxBits;
            int shift;
            float mul;
            if (Math.Abs(value) > 1)
            {
                int mulShift;
                mul = C.math.frexp(value, out mulShift);
                shift = Math.Min(maxShift, bits - mulShift);
                mul = mul * MathF.Pow(2, shift + mulShift);
            }
            else if (value == 0)
            {
                mul = shift = 0;
            }
            else
            {
                int mulShift;
                mul = C.math.frexp(value, out mulShift);
                shift = Math.Min(maxShift + mulShift, bits);
                mul = mul * MathF.Pow(2, shift);
                shift -= mulShift;
            }

            Debug.Assert(Math.Abs(mul) < MathF.Pow(2, bits));
            Debug.Assert(shift >= 0 && shift <= maxShift);
            Debug.Assert(Math.Abs(value - mul * MathF.Pow(2, -shift)) <= _floatEpsilon);
            return new FixedMul { Mul = mul, Shift = shift };
        }

        private ValueRange<float> Combine(in ValueRange<float> lhs, in ValueRange<float> rhs)
        {
            return new ValueRange<float>
            {
                Min = Math.Min(lhs.Min, rhs.Min),
                Max = Math.Max(lhs.Max, rhs.Max)
            };
        }
    }
}
