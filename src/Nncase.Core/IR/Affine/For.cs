// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public sealed class For : Expr
{
    public For(int memoryLevel, AffineMap domain, Expr body)
        : base(new Expr[] { domain, body })
    {
        MemoryLevel = memoryLevel;
    }

    public int Rank => Domain.Results.Length;

    public int MemoryLevel { get; }

    public AffineMap Domain => (AffineMap)Operands[0];

    public Expr Body => Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitFor(this, context);

    public For With(int? memoryLevel = null, AffineMap? domain = null, Expr? body = null)
        => new For(memoryLevel ?? MemoryLevel, domain ?? Domain, body ?? Body);
}
