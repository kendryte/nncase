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
/// unroll loop.
/// </summary>
public sealed class UnFoldBlock : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafBlock(Block expr)
    {
        if (expr.Predicate is TensorConst tc && tc.Value.ToScalar<bool>() is var predicate)
        {
            if (predicate)
            {
                return T.Sequential(expr.InitBody, expr.Body);
            }
            else
            {
                return T.Sequential().Build();
            }
        }

        return expr;
    }
}
