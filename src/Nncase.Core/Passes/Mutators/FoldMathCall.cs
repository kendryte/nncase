// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.Passes.Mutators;

/// <summary>
/// fold math calc operator.
/// </summary>
public sealed class FoldMathCall : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafCall(Call expr)
    {
        if (expr.Target is Op op && op.GetType().Namespace is string @namespace
          && @namespace.StartsWith("Nncase.IR.Math"))
        {
            return expr.Arguments.AsValueEnumerable().All(x => x is Const)
                ? Const.FromValue(CompilerServices.Evaluate(expr))
                : expr;
        }

        return expr;
    }
}
