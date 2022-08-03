// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.K210;

internal static class KPUUtility
{
    public static bool IsSupportedShape(Shape inShape)
    {
        return inShape.Count == 4
            && inShape[1].HasFixedValue(c => c <= 1024)
            && inShape[2].HasFixedValue(h => h is >= 4 and <= 256)
            && inShape[3].HasFixedValue(w => w is >= 4 and <= 512);
    }

    public static bool TryGetFilterType(int filterH, int filterW, out KPUFilterType filterType)
    {
        switch (filterH, filterW)
        {
            case (1, 1):
                filterType = KPUFilterType.Filter_1x1;
                return true;
            case (3, 3):
                filterType = KPUFilterType.Filter_3x3;
                return true;
            default:
                filterType = default;
                return false;
        }
    }

    public static int GetKPUPadding(KPUFilterType filterType)
    {
        return filterType switch
        {
            KPUFilterType.Filter_1x1 => 0,
            KPUFilterType.Filter_3x3 => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(filterType)),
        };
    }

    public static int GetKPUFilter(KPUFilterType filterType)
    {
        return filterType switch
        {
            KPUFilterType.Filter_1x1 => 1,
            KPUFilterType.Filter_3x3 => 3,
            _ => throw new ArgumentOutOfRangeException(nameof(filterType)),
        };
    }

    public static int[] GetPrePadding(ReadOnlySpan<int> paddings)
    {
        return new[] { paddings[0] > 0 ? paddings[0] : 0, paddings[1] > 0 ? paddings[1] : 0 };
    }

    public static int[] GetPostPadding(ReadOnlySpan<int> paddings)
    {
        return new[] { paddings[0] < 0 ? paddings[0] : 0, paddings[1] < 0 ? paddings[1] : 0 };
    }

    public static bool IsDepthWise(Expr conv2d, Expr input, int groups)
    {
        return input.CheckedShape[1].FixedValue == conv2d.CheckedShape[1].FixedValue
               && conv2d.CheckedShape[1].FixedValue == groups;
    }

    public static FakeKPUActivationParameters ClampToActivation(float[] clamp)
    {
        return new FakeKPUActivationParameters { Clamp = new ValueRange<float> { Min = clamp[0], Max = clamp[1] } };
    }

    public static KPUActivationParameters Activation()
    {
        return new KPUActivationParameters();
    }

    public static KPUBatchNormParameters BatchNorm()
    {
        return new KPUBatchNormParameters();
    }

    public static long carryShift(long value, Int32 shift)
    {
        if (shift > 0)
        {
            var integral = value >> shift;
            var fractional = value & (((1) <<shift)-1);
            var sign = value < 0 ? -1 : 1;
            var half = (((1) << 1) - 1) << (shift - 1);
            if (fractional<half)
            {
                return integral;
            }
            else if(fractional>half)
            {
                return integral + sign;
            }
            else
            {
                if ((integral & 1)!=0)
                    return integral + sign;
                // even
                else
                    return integral;
            }

            return value;
        }
        else if(shift<0)
        {
            value = value << (-shift);
        }

        return value;
    }
}
