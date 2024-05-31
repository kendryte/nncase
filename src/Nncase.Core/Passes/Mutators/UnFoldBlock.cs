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
                if (expr.AllocBuffers.Length > 0)
                {
                    var lets = expr.AllocBuffers.ToArray().Select(b => (T.Let(out var v, IR.F.Buffer.AllocateBufferView(b), b.Name), v)).ToArray();
                    for (int i = 0; i < lets.Length - 1; i++)
                    {
                        lets[i].Item1.Body(lets[i + 1].Item1);
                    }

                    var map = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
                    for (int i = 0; i < expr.AllocBuffers.Length; i++)
                    {
                        map.Add(expr.AllocBuffers[i], lets[i].v);
                    }

                    var mutator = new Substitutor(e =>
                    {
                        if (map.TryGetValue(e, out var r))
                        {
                            return r;
                        }

                        return null;
                    });

                    var initBody = mutator.Visit(expr.InitBody, Unit.Default);
                    var body = mutator.Visit(expr.Body, Unit.Default);

                    lets[^1].Item1.Body(initBody, body);
                    return lets[0].Item1.Build();
                }

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
