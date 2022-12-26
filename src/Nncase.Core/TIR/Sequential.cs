// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR.Builders;

namespace Nncase.TIR;

/// <summary>
/// The container of Exprs.
/// Represent a sequence of Expr.
/// </summary>
public sealed record Sequential(IRArray<Expr> Fields = default) : Expr, IReadOnlyList<Expr>
{
    /// <inheritdoc/>
    public int Count => Fields.Count;

    /// <summary>
    /// get the fields.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public Expr this[int index] => Fields[index];

    public static Sequential Flatten(IEnumerable<object> exprOrBuilders)
    {
        var ret = new List<Expr>();
        foreach (var item in exprOrBuilders)
        {
            Flatten(ret, item);
        }

        return new Sequential(new IRArray<Expr>(ret));
    }

    private static void Flatten(List<Expr> exprs, object exprOrBuilder)
    {
        switch (exprOrBuilder)
        {
            case Sequential sub:
                exprs.AddRange(Flatten(sub));
                break;
            case Expr expr:
                if (expr is not Call { Target: TIR.Nop })
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

    /// <inheritdoc/>
    public IEnumerator<Expr> GetEnumerator()
    {
        return Fields.GetEnumerator();
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
