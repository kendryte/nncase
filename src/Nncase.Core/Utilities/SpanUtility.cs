// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Utilities;

public static class SpanUtility
{
    public static ReadOnlySpan<TTo> UnsafeCast<TFrom, TTo>(ReadOnlySpan<TFrom> froms)
        where TFrom : class
        where TTo : class
    {
        ref var first = ref MemoryMarshal.GetReference(froms);
        ref var castFirst = ref Unsafe.As<TFrom, TTo>(ref first);
        return MemoryMarshal.CreateReadOnlySpan(ref castFirst, froms.Length);
    }
}
