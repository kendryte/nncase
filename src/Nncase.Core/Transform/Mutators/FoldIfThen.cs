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
/// fold if then and select.
/// </summary>
public sealed class FoldIfThen : ExprMutator
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

    public override Expr MutateLeaf(Call expr)
    {
        if (expr is { Target: IR.Math.Select } && expr[IR.Math.Select.Predicate] is TensorConst tc)
        {
            var c = tc.Value.ToScalar<bool>();
            return c ? expr[IR.Math.Select.TrueValue] : expr[IR.Math.Select.FalseValue];
        }

        return expr;
    }
}
