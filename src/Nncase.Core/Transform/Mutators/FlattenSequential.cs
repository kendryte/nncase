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
/// flatten sequential
/// </summary>
internal sealed class FlattenSequential : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(TIR.Sequential expr)
    {
        var flattened = TIR.Sequential.Flatten(expr);
        if (flattened.Count != expr.Count)
            return new TIR.Sequential() with { Fields = new(flattened) };
        return expr;
    }
}