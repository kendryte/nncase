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
/// unroll loop.
/// </summary>
public sealed class UnFoldBlock : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(TIR.Block expr)
    {
        if (expr.Predicate is TensorConst tc && tc.Value.ToScalar<bool>() is var predicate)
        {
            if (predicate)
            {
                return T.Sequential(Visit(expr.InitBody), Visit(expr.Body));
            }
            else
            {
                return T.Sequential().Build();
            }
        }

        return expr;
    }
}
