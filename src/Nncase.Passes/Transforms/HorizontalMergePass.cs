// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes.Rules;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Shape inference.
/// </summary>
public sealed class HorizontalMergePass : FunctionPass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="HorizontalMergePass"/> class.
    /// </summary>
    public HorizontalMergePass()
    {
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction pre, RunPassContext options)
    {
        if (pre is Function function)
        {
            var rewriter = new HorizontalMergeRewriter();
            return Task.FromResult((BaseFunction)rewriter.Rewrite(function));
        }

        return Task.FromResult(pre);
    }

    private sealed class HorizontalMergeRewriter : ExprRewriter
    {
        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (expr.Target is MatMul { OutputDataType: var outputDataType })
            {
                var aShape = (RankedShape)expr.CheckedShape;
                var rank = aShape.Rank;
                var input = (Expr)expr[MatMul.Lhs];

                Dimension cntNOffset = Dimension.Zero;
                var allMatMuls = new List<(Call Call, Expr W, Dimension NOffset, Dimension N)>
                {
                    (expr, (Expr)expr[MatMul.Rhs], cntNOffset, aShape[^1]),
                };
                cntNOffset += aShape[^1];

                foreach (var otherUser in input.Users.OfType<Call>())
                {
                    if (!ReferenceEquals(otherUser, expr) && otherUser.Target is MatMul matMulB && matMulB.OutputDataType == outputDataType)
                    {
                        var bShape = (RankedShape)otherUser.CheckedShape;
                        if (bShape.Rank == rank && bShape[..^1].SequenceEqual(aShape[..^1]))
                        {
                            // If the shapes match, we can merge them
                            var n = bShape[^1];
                            allMatMuls.Add((otherUser, (Expr)otherUser[MatMul.Rhs], cntNOffset, n));
                            cntNOffset += n;
                        }
                    }
                }

                if (allMatMuls.Count > 1)
                {
                    var newAllMatMul = IR.F.Math.MatMul(input, IR.F.Tensors.Concat(new IR.Tuple(allMatMuls.Select(x => x.W).ToArray()), rank - 1), outputDataType);

                    foreach (var (call, _, nOffset, n) in allMatMuls)
                    {
                        var newMatMul = IR.F.Tensors.Slice(newAllMatMul, [nOffset], [nOffset + n], [rank - 1], [1]);
                        newMatMul.InheritMetaData(call);
                        ReplaceUtility.ReplaceAllUsesWith(call, newMatMul);
                        ExprMemo[call] = newMatMul;
                    }

                    return ExprMemo[expr];
                }
            }

            return expr;
        }
    }
}
