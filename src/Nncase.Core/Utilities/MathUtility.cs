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
        where T : INumber<T> => (value + (align - T.One)) / align * align;

    public static T CeilDiv<T>(T value, T div)
        where T : INumber<T> => (value + (div - T.One)) / div;

    public static int Factorial(int n)
    {
        if (n == 0)
        {
            return 1;
        }

        return n * Factorial(n - 1);
    }
}
