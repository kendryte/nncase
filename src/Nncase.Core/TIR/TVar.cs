// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using System;
namespace Nncase.TIR
{
    /// <summary>
    /// Tensor Range Define.
    /// </summary>
    /// <param name="Min">beginning of the nodes.</param>
    /// <param name="Max">the extend of range.</param>
    public sealed record Range(Expr Min, Expr Max)
    {
        /// <summary>
        /// <see cref="Range"/>.
        /// </summary>
        /// <param name="tuple"> value tuple. </param>
        public static implicit operator Range((Expr, Expr) tuple) => new Range(tuple.Item1, tuple.Item2);

        /// <summary>
        /// <see cref="Range"/>.
        /// </summary>
        /// <param name="End">end expr.</param>
        public static implicit operator Range(Expr End) => new Range(0, End);
    }

    /// <summary>
    ///  Iteration Variable like a symobl, It represents an iteration over an integer interval.
    /// </summary>
    /// <param name="TypeAnnotation">The Type Annotation.</param>
    /// <param name="Dom">
    ///  the domain of iteration, if known, can be None For the intermediate schedule node, before schedule.
    /// </param>
    /// <param name="Mode">The type of the IterVar. </param>
    /// <param name="Value">The looping variable. </param>
    public sealed record IterVar(IRType TypeAnnotation, Range Dom, IterationMode Mode, Expr Value) : Expr
    {
    }
}