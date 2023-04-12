// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Tensor Range Define.
/// </summary>
public sealed partial class Range : Expr
{
    /// <summary>
    /// the full range.
    /// </summary>
    public static readonly Range All = new Range(int.MinValue, int.MaxValue, 1);

    public Range(Expr start, Expr stop, Expr step)
        : base(new[] { start, stop, step })
    {
    }

    /// <summary>
    /// Gets beginning of the nodes.
    /// </summary>
    public Expr Start => Operands[0];

    /// <summary>
    /// Gets stop of the nodes.
    /// </summary>
    public Expr Stop => Operands[1];

    /// <summary>
    /// Gets the extend of range.
    /// </summary>
    public Expr Step => Operands[2];

    /// <summary>
    /// <see cref="Range"/>.
    /// </summary>
    public static implicit operator Range(System.Range range)
    {
        if (range.Equals(System.Range.All))
        {
            return All;
        }

        if (range.Start.IsFromEnd || range.End.IsFromEnd)
        {
            throw new NotSupportedException("The System.Range From End.");
        }

        return new Range(range.Start.Value, range.End.Value, 1);
    }

    public static Range operator *(Range range, Expr expr) => new Range(range.Start * expr, range.Stop * expr, range.Step);

    public static Range operator -(Range range, Expr expr) => new Range(range.Start - expr, range.Stop - expr, range.Step);

    public static Range operator +(Range range, Expr expr) => new Range(range.Start + expr, range.Stop + expr, range.Step);

    public static Range operator /(Range range, Expr expr) => new Range(range.Start / expr, range.Stop / expr, range.Step);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitRange(this, context);

    public Range With(Expr? start = null, Expr? stop = null, Expr? step = null)
        => new Range(start ?? Start, stop ?? Stop, step ?? Step);
}
