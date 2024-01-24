// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Affine;

public sealed class Load : Expr
{
    public Load(Expr source, AffineMap region)
        : base(new[] { source, region })
    {
    }

    public Expr Source => Operands[0];

    public AffineMap Region => (AffineMap)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitLoad(this, context);
}
