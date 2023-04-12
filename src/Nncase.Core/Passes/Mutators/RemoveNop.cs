// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// remove the nop call from the body.
/// </summary>
public sealed class RemoveNop : ExprRewriter
{
    protected override Expr RewriteLeafSequential(Sequential expr)
    {
        bool mutated = false;
        var body = new List<Expr>();
        foreach (var item in expr.Fields)
        {
            if (item is not Call { Target: Nop })
            {
                body.Add(item);
            }
            else
            {
                mutated = true;
            }
        }

        return mutated ? expr.With(fields: body.ToArray()) : expr;
    }
}
