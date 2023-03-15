// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

// internal enum Inst

/// <summary>
/// substitutor will not substitute the other function.
/// </summary>
public sealed class Substitutor : ExprRewriter
{
    private readonly Func<Expr, Expr?> _mapper;

    public Substitutor(Func<Expr, Expr?> maper)
        : base(false)
    {
        _mapper = maper;
    }

    /// <inheritdoc/>
    protected override Expr DefaultRewriteLeaf(Expr expr)
    {
        if (_mapper(expr) is Expr replace)
        {
            return replace;
        }

        return expr;
    }
}
