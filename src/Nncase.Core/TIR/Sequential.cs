// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Nncase.Collections;
using Nncase.IR;
using Nncase.TIR.Builders;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// The container of Exprs.
/// Represent a sequence of Expr.
/// </summary>
public sealed class Sequential : Expr
{
    public static readonly Sequential Empty = new Sequential(ReadOnlySpan<Expr>.Empty);

    public Sequential(ReadOnlySpan<Expr> fields)
        : base(fields.ToArray())
    {
    }

    public Sequential(params Expr[] fields)
        : base(fields.ToArray())
    {
    }

    public ReadOnlySpan<Expr> Fields => Operands;

    public int Count => Fields.Length;

    /// <summary>
    /// get the fields.
    /// </summary>
    public Expr this[int index] => Fields[index];

    public static Sequential Flatten(ReadOnlySpan<object> exprOrBuilders)
    {
        var ret = new List<Expr>();
        foreach (var item in exprOrBuilders)
        {
            Flatten(ret, item);
        }

        return new Sequential(CollectionsMarshal.AsSpan(ret));
    }

    public static Sequential Flatten(ReadOnlySpan<Expr> exprs) => Flatten(SpanUtility.UnsafeCast<Expr, object>(exprs));

    public static Sequential Flatten(Expr[] exprs) => Flatten(exprs.AsSpan());

    public static Sequential Flatten(object[] exprs) => Flatten(exprs.AsSpan());

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitSequential(this, context);

    public Sequential With(Expr[]? fields = null) => new Sequential(fields ?? Fields);

    private static void Flatten(List<Expr> exprs, object exprOrBuilder)
    {
        switch (exprOrBuilder)
        {
            case Sequential sub:
                exprs.AddRange(Flatten(sub.Fields).Fields);
                break;
            case Expr expr:
                if (expr is not Call { Target: Nop })
                {
                    exprs.Add(expr);
                }

                break;
            case IExprBuilder<Expr> builder:
                Flatten(exprs, builder.Build());
                break;
            default:
                throw new ArgumentException("Invalid exprOrBuilder type: " + exprOrBuilder.ToString());
        }
    }
}
