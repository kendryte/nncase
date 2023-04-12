// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using SMath = System.Math;

namespace Nncase.Utilities;

/// <summary>
/// Array utility.
/// </summary>
public static class QuantUtility
{
    public enum AdaMode
    {
        /// <summary>
        /// Conv2D.
        /// </summary>
        Conv2D,

        /// <summary>
        /// Conv2DTranspose.
        /// </summary>
        Conv2DTranspose,

        /// <summary>
        /// Linear.
        /// </summary>
        Linear,
    }

    /// <summary>
    /// GetQuantParam.
    /// </summary>
    public static QuantParam GetQuantParam(ValueRange<float> range, int bits, QuantMode quantMode)
    {
        range = FixupRange(range, quantMode == QuantMode.SignedSymmetricMode);
        double qMax;
        double qMin;
        switch (quantMode)
        {
            case QuantMode.UnsignedMode:
                qMin = 0;
                qMax = (1 << bits) - 1;
                break;
            case QuantMode.SignedSymmetricMode:
                qMin = -(1 << (bits - 1)) + 1;
                qMax = (1 << (bits - 1)) - 1;
                break;
            case QuantMode.SignedAsymmetricMode:
                qMin = -(1 << (bits - 1));
                qMax = (1 << (bits - 1)) - 1;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(quantMode), "Invalid QuantMode");
        }

        var scale = (range.Max - range.Min) / (qMax - qMin);
        var bias = SMath.Round(range.Min * (qMin - qMax) / (range.Max - range.Min)) + qMin;
        return new QuantParam((int)bias, (float)scale);
    }

    /// <summary>
    /// fixup range.
    /// </summary>
    public static ValueRange<float> FixupRange(ValueRange<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            var r = SMath.Max(SMath.Max(SMath.Abs(range.Min), SMath.Abs(range.Max)), 0.01f);
            return (-r, r);
        }
        else
        {
            range.Max = SMath.Max(0, range.Max);
            range.Min = SMath.Min(0, range.Min);
            var r = range.Max - range.Min;
            if (r == 0)
            {
                r = 0.1f;
            }
            else if (r < 0.01f)
            {
                r = 0.01f;
            }

            range.Max = range.Min + r;
        }

        return range;
    }

    public static ValueRange<T> GetRange<T>(Span<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var data = input.ToArray();
        return new(data.Min(), data.Max());
    }

    public static List<float> GetWeightsRangesByChannel(Span<float> weights, int channels)
    {
        var tmpMin = float.MaxValue;
        var tmpMax = float.MinValue;
        var minMaxArr = new List<float>();
        for (int i = 0; i < weights.Length; i++)
        {
            if (i % (weights.Length / channels) == 0)
            {
                tmpMin = float.MaxValue;
                tmpMax = float.MinValue;
            }

            if (weights[i] < tmpMin)
            {
                tmpMin = weights[i];
            }

            if (weights[i] > tmpMax)
            {
                tmpMax = weights[i];
            }

            if ((i + 1) % (weights.Length / channels) == 0)
            {
                tmpMax = Math.Max(0, tmpMax);
                tmpMin = Math.Min(0, tmpMin);
                var r = tmpMax - tmpMin;
                if (r == 0)
                {
                    r = 0.1f;
                }
                else if (r < 0.01f)
                {
                    r = 0.01f;
                }

                tmpMax = tmpMin + r;

                minMaxArr.Add(tmpMin);
                minMaxArr.Add(tmpMax);
            }
        }

        return minMaxArr;
    }
}
