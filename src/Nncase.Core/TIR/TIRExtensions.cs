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
/// TIRExtensions.
/// </summary>
public static class TIRExtensions
{
    /// <summary>
    /// Get the tir op buffer allocation reuse information.
    /// </summary>
    /// <returns> map dest index to source index. </returns>
    public static Dictionary<int, int> GetInPlaceMemo(this Op op)
    {
        return op.GetType().GetCustomAttributes(typeof(ParameterInPlaceAttribute), true).OfType<ParameterInPlaceAttribute>().ToDictionary(a => a.DestIndex, a => a.SourceIndex);
    }

    /// <summary>
    /// convert IEnumerable to tir Sequential.
    /// </summary>
    /// <param name="enumerable"> instance.</param>
    /// <returns> Sequential. </returns>
    public static Sequential ToSequential(this IEnumerable<Expr> enumerable) => new Sequential(enumerable.ToArray());
}
