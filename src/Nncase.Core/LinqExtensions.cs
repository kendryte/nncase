// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Linq extensions.
/// </summary>
public static class LinqExtensions
{
    /// <summary>
    /// Get cartesian product.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="sequences">Source sequences.</param>
    /// <returns>Cartesian product.</returns>
    public static IEnumerable<IEnumerable<T>> CartesianProduct<T>(this IEnumerable<IEnumerable<T>> sequences)
    {
        IEnumerable<IEnumerable<T>> emptyProduct = new[] { Enumerable.Empty<T>() };
        return sequences.Aggregate(
            emptyProduct,
            (accumulator, sequence) =>
              from accseq in accumulator
              from item in sequence
              select accseq.Concat(new[] { item }));
    }

    /// <summary>
    /// take or default.
    /// </summary>
    public static IEnumerable<T> TakeOrDefault<T>(this IEnumerable<T> items, int count, T defaultValue)
    {
        if (items.Count() < count)
        {
            return items.Concat(Enumerable.Repeat(defaultValue, count - items.Count()));
        }

        return items;
    }
}
