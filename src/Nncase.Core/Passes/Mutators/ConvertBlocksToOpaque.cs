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
/// Substitute all the block vars with the PrimExprs they are bound to, indicated by the corresponding iter_values in BlockRealize, for opaque blocks by removing all . the iter_values in BlockRealize and iter_vars in Block.
/// </summary>
public sealed class ConvertBlocksToOpaque : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafIterVar(IterVar expr)
    {
        return expr.Value;
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafBlock(Block expr)
    {
        return expr.With(
            iterVars: Array.Empty<IterVar>());
    }
}
