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
/// substitutor will not substitute the other function.
/// </summary>
public sealed class Substitutor : ExprMutator
{
    private readonly Func<Expr, Expr?> _maper;

    public Substitutor(Func<Expr, Expr?> maper)
    {
        _maper = maper;
    }

    /// <inheritdoc/>
    public override Expr DefaultMutateLeaf(Expr expr)
    {
        var mexpr = _maper(expr);
        if (mexpr is not null)
        {
            return mexpr;
        }

        return expr;
    }
}
