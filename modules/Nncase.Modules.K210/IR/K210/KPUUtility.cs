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


}
