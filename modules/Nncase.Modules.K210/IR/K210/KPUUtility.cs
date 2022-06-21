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

    public static int[] GetPrePadding(ReadOnlySpan<int> paddings)
    {
        return new[] { paddings[0] > 0 ? paddings[0] : 0, paddings[1] > 0 ? paddings[1] : 0 };
    }

    public static int[] GetPostPadding(ReadOnlySpan<int> paddings)
    {
        return new[] { paddings[0] < 0 ? paddings[0] : 0, paddings[1] < 0 ? paddings[1] : 0 };
    }

    public static FakeKPUActivationParameters ClampToActivation(float[] clamp)
    {
        return new FakeKPUActivationParameters { Clamp = new ValueRange<float> { Min = clamp[0], Max = clamp[1] } };
    }
}
