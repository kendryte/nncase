// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Collections;

public static class CollectionsExtensions
{
    public static void AddRange<T>(this List<T> list, ReadOnlySpan<T> items)
    {
        list.Capacity += items.Length;
        foreach (var item in items)
        {
            list.Add(item);
        }
    }
}
