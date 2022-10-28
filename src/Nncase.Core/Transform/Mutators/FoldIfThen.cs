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
internal sealed class FoldIfThen  : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(TIR.IfThenElse expr)
    {
        if (expr.Condition is TensorConst { Value: Tensor<bool> value })
        {
            return value.ToScalar() ? expr.Then : expr.Else;
        }
        return expr;
    }
}