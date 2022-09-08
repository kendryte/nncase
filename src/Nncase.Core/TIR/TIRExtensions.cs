// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// TIRExtensions
/// </summary>
public static class TIRExtensions
{
    /// <summary>
    /// convert IEnumerable to tir Sequential.
    /// </summary>
    /// <param name="enumerable"> instance.</param>
    /// <returns> Sequential. </returns>
    public static Sequential ToSequential(this IEnumerable<Expr> enumerable) => new Sequential(new IRArray<Expr>(enumerable));

    /// <summary>
    /// get the total elements bytes count.
    /// </summary>
    // public static int SizeInBytes<T>(this IBufferView<T> view)
    //   where T : class
    //   => view.Size() * view.DType.SizeInBytes;

    /// <summary>
    /// get the total elements.
    /// </summary>
    // public static int Size<T>(this IBufferView<T> view)
    //   where T : class
    // => view.Shape.ToArray().Aggregate(1, (acc, dim) => acc * dim);


    /// <summary>
    /// print the tensorview string
    /// </summary>
    /// <typeparam name="T">type</typeparam>
    /// <param name="view">view.</param>
    /// <returns></returns>
    public static string View<T>(this IBufferView<T> view)
      where T : class
    => $"[{string.Join(", ", view.SelectedRanges.ToArray().Select(rg => $"{rg.Start}:{rg.End}"))}]";
}