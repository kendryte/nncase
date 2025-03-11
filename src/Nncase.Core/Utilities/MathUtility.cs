// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Utilities;

public static class MathUtility
{
    public static T AlignUp<T>(T value, T align)
        where T : IBinaryInteger<T> => (value + (align - T.One)) / align * align;

    public static T CeilDiv<T>(T value, T div)
        where T : IBinaryInteger<T> => (value + (div - T.One)) / div;

    public static float CeilDiv(float value, float div) => MathF.Ceiling(value / div);

    public static double CeilDiv(double value, double div) => Math.Ceiling(value / div);

    public static int Factorial(int n)
    {
        if (n == 0)
        {
            return 1;
        }

        return n * Factorial(n - 1);
    }
}
