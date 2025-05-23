// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// Return output.
/// </summary>
public sealed class Return : Expr
{
    public Return(ReadOnlySpan<Expr> values)
        : base(values.AsValueEnumerable().Select(x => (BaseExpr)x).ToArray())
    {
    }

    /// <summary>
    /// Gets the value.
    /// </summary>
    public ReadOnlySpan<Expr> Values => SpanUtility.UnsafeCast<BaseExpr, Expr>(Operands);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitReturn(this, context);

    public Return With(Expr[]? values = null)
        => new Return(values ?? Values);
}
