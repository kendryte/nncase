// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Utilities;

public static class LinqUtility
{
    public static IEnumerable<T> Range<T>(T start, int count)
        where T : INumber<T>
    {
        for (int i = 0; i < count; i++)
        {
            yield return start++;
        }
    }
}
