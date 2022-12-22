// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Transform.Mutators;

/// <summary>
/// fold math calc operator.
/// </summary>
internal sealed class FoldMathCall : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(Call expr)
    {
        if (expr.Target is Op op && op.GetType().Namespace is string _namespace
          && _namespace.StartsWith("Nncase.IR.Math"))
        {
            return expr.Parameters.Select(Visit).All(e => e is Const) ? StructEqualFolding(Const.FromValue(CompilerServices.Evaluate(expr))) : expr;
        }

        return expr;
    }
}
