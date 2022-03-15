// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
namespace Nncase.TIR;

/// <summary>
/// Tensor Range Define.
/// </summary>
/// <param name="Start">beginning of the nodes.</param>
/// <param name="Stop">.</param>
/// <param name="Step">the extend of range.</param>
public sealed record Range(Expr Start, Expr Stop, Expr Step)
{
    /// <summary>
    /// <see cref="Range"/>.
    /// </summary>
    /// <param name="tuple"> value tuple. </param>
    public static implicit operator Range((Expr, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, 1);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((Expr, Expr, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>.
    /// </summary>
    /// <param name="Stop">end expr.</param>
    public static implicit operator Range(Expr Stop) => new Range(0, Stop, 1);
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
