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
/// Substitute all the block vars with the PrimExprs they are bound to, indicated by the corresponding iter_values in BlockRealize, for opaque blocks by removing all . the iter_values in BlockRealize and iter_vars in Block.
/// </summary>
public sealed class ConvertBlocksToOpaque : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(IterVar expr)
    {
        return expr.Value;
    }

    /// <inheritdoc/>
    public override Expr MutateLeaf(Block expr)
    {
        return expr with
        {
            // the block realize
            InitBody = (TIR.Sequential)Visit(expr.InitBody),
            Predicate = Visit(expr.Predicate),
            IterVars = new(),

            // the block internal.
            Body = (TIR.Sequential)Visit(expr.Body),
            Reads = MutateArray(expr.Reads, b => (BufferRegion)Visit(b)),
            Writes = MutateArray(expr.Writes, b => (BufferRegion)Visit(b)),
        };
    }
}
