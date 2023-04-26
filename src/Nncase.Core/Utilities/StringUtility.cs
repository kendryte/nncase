// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.TIR;
using static NetFabric.Hyperlinq.ArrayExtensions;

namespace Nncase.Utilities;

public static class StringUtility
{
    public static string Join<TSource, TResult, TSelector>(ReadOnlySpan<char> separator, in SpanSelectEnumerable<TSource, TResult, TSelector> values)
            where TSelector : struct, IFunction<TSource, TResult>
    {
        var en = values.GetEnumerator();
        if (!en.MoveNext())
        {
            return string.Empty;
        }

        // We called MoveNext once, so this will be the first item
        var currentValue = en.Current;

        // Call ToString before calling MoveNext again, since
        // we want to stay consistent with the below loop
        // Everything should be called in the order
        // MoveNext-Current-ToString, unless further optimizations
        // can be made, to avoid breaking changes
        string? firstString = currentValue?.ToString();

        // If there's only 1 item, simply call ToString on that
        if (!en.MoveNext())
        {
            // We have to handle the case of either currentValue
            // or its ToString being null
            return firstString ?? string.Empty;
        }

        var result = new StringBuilder();

        result.Append(firstString);

        do
        {
            currentValue = en.Current;

            result.Append(separator);
            if (currentValue != null)
            {
                result.Append(currentValue.ToString());
            }
        }
        while (en.MoveNext());

        return result.ToString();
    }

#if false
    // “WhereSelect” constructor method is modified by private
    public static string Join<TSource, TResult, TPredicate, TSelector>(ReadOnlySpan<char> separator, in SpanWhereSelectEnumerable<TSource, TResult, TPredicate, TSelector> values)
            where TSelector : struct, IFunction<TSource, TResult>
            where TPredicate : struct, IFunction<TSource, bool>
    {
        var en = values.GetEnumerator();
        if (!en.MoveNext())
        {
            return string.Empty;
        }

        // We called MoveNext once, so this will be the first item
        var currentValue = en.Current;

        // Call ToString before calling MoveNext again, since
        // we want to stay consistent with the below loop
        // Everything should be called in the order
        // MoveNext-Current-ToString, unless further optimizations
        // can be made, to avoid breaking changes
        string? firstString = currentValue?.ToString();

        // If there's only 1 item, simply call ToString on that
        if (!en.MoveNext())
        {
            // We have to handle the case of either currentValue
            // or its ToString being null
            return firstString ?? string.Empty;
        }

        var result = new StringBuilder();

        result.Append(firstString);

        do
        {
            currentValue = en.Current;

            result.Append(separator);
            if (currentValue != null)
            {
                result.Append(currentValue.ToString());
            }
        }
        while (en.MoveNext());

        return result.ToString();
    }
#endif
}
