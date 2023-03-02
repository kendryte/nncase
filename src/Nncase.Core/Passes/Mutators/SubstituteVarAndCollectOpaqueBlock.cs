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

/// <summary>
/// Substitute vars and collect the reuse mapping of opaque blocks.
/// </summary>
public sealed class SubstituteVarAndCollectOpaqueBlock : ExprRewriter
{
    private readonly Func<Var, Expr?> _varMapper;
    private readonly Dictionary<Block, Block> _opaqueBlocks;

    /// <summary>
    /// Initializes a new instance of the <see cref="SubstituteVarAndCollectOpaqueBlock"/> class.
    /// <see cref="SubstituteVarAndCollectOpaqueBlock"/>.
    /// </summary>
    public SubstituteVarAndCollectOpaqueBlock(
        Func<Var, Expr?> varMaper,
        Dictionary<Block, Block> opaque_blocks)
    {
        _varMapper = varMaper;
        _opaqueBlocks = opaque_blocks;
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafVar(Var expr)
    {
        if (_varMapper(expr) is Expr replace)
        {
            return replace;
        }

        return expr;
    }

    protected override Expr RewriteLeafBlock(Block expr)
    {
        var replace = (Block)base.RewriteLeafBlock(expr);
        if (replace.IterVars.IsEmpty)
        {
            _opaqueBlocks.Add(expr, replace);
        }

        return replace;
    }
}
