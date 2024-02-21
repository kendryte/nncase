// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Linq extensions.
/// </summary>
public static class LinqExtensions
{
    /// <summary>
    /// Get the ranges from range desc.
    /// </summary>
    /// <param name="stride">stride.</param>
    /// <param name="start">start.</param>
    /// <param name="stop">stop.</param>
    /// <returns>Ranges.</returns>
    public static IEnumerable<Range> Ranges(this int stride, int start, int stop)
    {
        for (int i = start; i < stop; i += stride)
        {
            yield return new Range(i, Math.Min(stop, i + stride));
        }
    }

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
    /// Get the permutation of the source.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="source">Source sequences.</param>
    /// <returns>Permutated sequences.</returns>
    public static IEnumerable<T[]> Permutate<T>(this IEnumerable<T> source)
    {
        return Permutation(source, Enumerable.Empty<T>());

        IEnumerable<T[]> Permutation(IEnumerable<T> reminder, IEnumerable<T> prefix) =>
            !reminder.Any() ? new[] { prefix.ToArray() } :
            reminder.SelectMany((c, i) => Permutation(
                reminder.Take(i).Concat(reminder.Skip(i + 1)).ToArray(),
                prefix.Append(c)));
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

    public static UInt128 Sum<TSource>(this IEnumerable<TSource> source, Func<TSource, UInt128> selector)
    {
        UInt128 acc = 0;
        foreach (var item in source)
        {
            acc += selector(item);
        }

        return acc;
    }

    public static UInt128 Sum(this IEnumerable<UInt128> source)
    {
        UInt128 acc = 0;
        foreach (var item in source)
        {
            acc += item;
        }

        return acc;
    }
}
