// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// remove calling function wrapper.
/// </summary>
public sealed class RemoveFunctionWrapper : ExprRewriter
{
    protected override BaseExpr RewriteLeafCall(Call expr)
    {
        if (expr.Target is FunctionWrapper { Target: PrimFunctionWrapper { Target: PrimFunction primFunc } })
        {
            var flattenArgs = expr.Arguments.ToArray()
                .SelectMany(x => x is IR.Tuple tuple ? tuple.Fields.ToArray() : [x])
                .ToArray();
            return expr.With(target: primFunc, arguments: flattenArgs);
        }

        return expr;
    }
}
