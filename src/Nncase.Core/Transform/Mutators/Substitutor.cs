// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform.Mutators;


// internal enum Inst

/// <summary>
/// substitutor will not substitute the other function
/// </summary>
internal sealed class Substitutor : ExprMutator
{
    Func<Expr, Expr?> Maper;
    bool Entered;

    public Substitutor(Func<Expr, Expr?> maper)
    {
        Maper = maper;
        Entered = false;
    }

    public Expr VisitBaseFunc(BaseFunction expr)
    {
        if (!Entered)
        {
            Entered = true;
            return expr switch
            {
                Function x => base.Visit(x),
                PrimFunction x => base.Visit(x),
                PrimFunctionWrapper x => base.Visit(x),
                Fusion x => base.Visit(x),
                _ => throw new NotSupportedException()
            };
        }
        return expr;
    }

    public override Expr Visit(Function expr) => VisitBaseFunc(expr);
    public override Expr Visit(PrimFunction expr) => VisitBaseFunc(expr);
    public override Expr Visit(PrimFunctionWrapper expr) => VisitBaseFunc(expr);
    public override Expr Visit(Fusion expr) => VisitBaseFunc(expr);

    /// <inheritdoc/>
    public override Expr DefaultMutateLeaf(Expr expr)
    {
        var mexpr = Maper(expr);
        if (mexpr is not null) { return mexpr; }
        return expr;
    }
}