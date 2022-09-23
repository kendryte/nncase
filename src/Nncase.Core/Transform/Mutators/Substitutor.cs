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

    public Substitutor(Func<Expr, Expr?> maper)
    {
        Maper = maper;
    }

    /// <inheritdoc/>
    public override Expr DefaultMutateLeaf(Expr expr)
    {
        var mexpr = Maper(expr);
        if (mexpr is not null) { return mexpr; }
        return expr;
    }
}