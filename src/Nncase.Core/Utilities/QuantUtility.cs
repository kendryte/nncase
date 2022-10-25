// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SMath = System.Math;

namespace Nncase.Utilities;

/// <summary>
/// Array utility.
/// </summary>
public static class QuantUtility
{
    /// <summary>
    /// GetQuantParam
    /// </summary>
    /// <param name="range"></param>
    /// <param name="bits"></param>
    /// <param name="quantMode"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static QuantParam GetQuantParam(ValueRange<float> range, int bits, QuantMode quantMode)
    {
        range = FixupRange(range, quantMode == QuantMode.SignedSymmetricMode);
        double QMax = 255;
        double QMin = 0;
        switch (quantMode)
        {
            case QuantMode.UnsignedMode:
                QMin = 0;
                QMax = (1 << bits) - 1;
                break;
            case QuantMode.SignedSymmetricMode:
                QMin = -(1 << (bits - 1)) + 1;
                QMax = (1 << (bits - 1)) - 1;
                break;
            case QuantMode.SignedAsymmetricMode:
                QMin = -(1 << (bits - 1));
                QMax = (1 << (bits - 1)) - 1;
                break;
            default:
                throw new ArgumentOutOfRangeException("Invalid QuantMode");
        }

        var scale = (range.Max - range.Min) / (QMax - QMin);
        var bias = SMath.Round((range.Min * (QMin - QMax)) / (range.Max - range.Min)) + QMin;
        return new QuantParam((int)bias, (float)scale);
    }

    /// <summary>
    /// fixup range.
    /// </summary>
    /// <param name="range"></param>
    /// <param name="symmetric"></param>
    /// <returns></returns>
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
                r = 0.1f;
            else if (r < 0.01f)
                r = 0.01f;
            range.Max = range.Min + r;
        }
        return range;
    }

    public static ValueRange<T> GetRange<T>(Span<T> input) where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var data = input.ToArray();
        return new(data.Min(), data.Max());
    }
}