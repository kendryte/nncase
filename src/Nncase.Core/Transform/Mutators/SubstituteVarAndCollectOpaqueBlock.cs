// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform.Mutators;

/// <summary>
/// Substitute vars and collect the reuse mapping of opaque blocks.
/// </summary>
internal sealed class SubstituteVarAndCollectOpaqueBlock : ExprMutator
{
    private readonly Func<Var, Expr?> _varMaper;
    private readonly Dictionary<Block, Block> _opaqueBlocks;

    /// <summary>
    /// Initializes a new instance of the <see cref="SubstituteVarAndCollectOpaqueBlock"/> class.
    /// <see cref="SubstituteVarAndCollectOpaqueBlock"/>.
    /// </summary>
    /// <param name="varMaper"></param>
    /// <param name="opaque_blocks"></param>
    public SubstituteVarAndCollectOpaqueBlock(
        Func<Var, Expr?> varMaper,
        Dictionary<Block, Block> opaque_blocks)
    {
        _varMaper = varMaper;
        _opaqueBlocks = opaque_blocks;
    }

    /// <inheritdoc/>
    public override Expr MutateLeaf(Var expr)
    {
        if (_varMaper(expr) is var nexpr && nexpr is not null)
        {
            IsMutated = true;
            return nexpr;
        }

        return expr;
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Block expr)
    {
        var nblock = (Block)base.VisitLeaf(expr);
        if (nblock.IterVars.Count == 0)
        {
            _opaqueBlocks.Add(expr, nblock);
        }

        return nblock;
    }
}
