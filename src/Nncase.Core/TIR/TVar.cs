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
public sealed record Range(Expr Start, Expr Stop, Expr Step) : IR.IMutatable
{
    /// <summary>
    /// <see cref="Range"/>.
    /// </summary>
    /// <param name="tuple"> value tuple. </param>
    public static implicit operator Range((Expr, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, 1);

    /// <summary>
    /// <see cref="Range"/>.
    /// </summary>
    /// <param name="tuple"> value tuple. </param>
    public static implicit operator Range((int, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, 1);

    /// <summary>
    /// <see cref="Range"/>.
    /// </summary>
    /// <param name="tuple"> value tuple. </param>
    public static implicit operator Range((int, int) tuple) => new Range(tuple.Item1, tuple.Item2, 1);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((Expr, Expr, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((int, Expr, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((Expr, int, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((Expr, Expr, int) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((int, int, Expr) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((int, Expr, int) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((Expr, int, int) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="tuple"></param>
    public static implicit operator Range((int, int, int) tuple) => new Range(tuple.Item1, tuple.Item2, tuple.Item3);

    /// <summary>
    /// <see cref="Range"/>
    /// </summary>
    /// <param name="range"></param>
    public static implicit operator Range(System.Range range)
    {
        if (range.Equals(System.Range.All))
            return All;
        if (range.Start.IsFromEnd || range.End.IsFromEnd)
            throw new NotSupportedException("The System.Range From End.");
        return new Range(range.Start.Value, range.End.Value, 1);
    }


    /// <summary>
    /// the full range.
    /// </summary>
    public static readonly Range All = new Range(int.MinValue, int.MaxValue, 1);

    /// <inheritdoc/>
    public object WithNew(ExprMutator mutator)
    {
        return new TIR.Range(mutator.Visit(Start), mutator.Visit(Stop), mutator.Visit(Step));
    }

    /// <inheritdoc/>
    public object Visit<TExprResult, TTypeResult>(ExprFunctor<TExprResult, TTypeResult> functor)
    {
        functor.Visit(Start);
        functor.Visit(Stop);
        functor.Visit(Step);
        return default(object)!;
    }

    /// <inheritdoc/>
    public static Range operator *(Range range, Expr expr) => new Range(range.Start * expr, range.Stop * expr, range.Step);

    /// <inheritdoc/>
    public static Range operator -(Range range, Expr expr) => new Range(range.Start - expr, range.Stop - expr, range.Step);

    /// <inheritdoc/>
    public static Range operator +(Range range, Expr expr) => new Range(range.Start + expr, range.Stop + expr, range.Step);

    /// <inheritdoc/>
    public static Range operator /(Range range, Expr expr) => new Range(range.Start / expr, range.Stop / expr, range.Step);
}

/// <summary>
///  Iteration Variable like a symobl, It represents an iteration over an integer interval.
/// </summary>
/// <param name="Dom">
///  the domain of iteration, if known, can be None For the intermediate schedule node, before schedule.
/// </param>
/// <param name="Mode">The type of the IterVar. </param>
/// <param name="Value">The looping variable. </param>
public sealed record IterVar(Range Dom, IterationMode Mode, Var Value) : Expr
{
}
