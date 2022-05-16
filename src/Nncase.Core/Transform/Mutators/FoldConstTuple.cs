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
/// unroll loop
/// </summary>
internal sealed class FoldConstTuple : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(IR.Tuple expr)
    {
        if (expr.Select(Visit).All(e => e is Const))
        {
            return new TupleConst(new(expr.Select(Visit).OfType<Const>()));
        }
        return expr;
    }
}