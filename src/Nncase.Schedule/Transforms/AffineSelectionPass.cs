// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

public abstract class AffineSelectionPass : FunctionPass
{
    public AffineSelectionPass(string moduleKind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input.ModuleKind == ModuleKind
            && input is Function func)
        {
            var rewriter = new AffineSelectionRewriter(this);
            var newFunc = (BaseFunction)rewriter.Rewrite(func);
            return Task.FromResult(newFunc);
        }

        return Task.FromResult(input);
    }

    protected abstract Expr SelectCall(Call call, Expr output);

    protected Expr SelectUnaryLike(Op op, Op tirOp, Call call, Expr output)
    {
        var input = call.Arguments[IR.Math.Unary.Input.Index];
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 })
        {
            return call;
        }

        if (op.Parameters.Count != 1)
        {
            throw new ArgumentException($"Unary-like op {op} should have only 1 parameter");
        }

        var rank = input.CheckedShape.Rank;
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(new Call(tirOp, inTile, outTile))
            .Build();
    }

    private sealed class AffineSelectionRewriter : ExprRewriter
    {
        private readonly AffineSelectionPass _selectionPass;

        public AffineSelectionRewriter(AffineSelectionPass selectionPass)
        {
            _selectionPass = selectionPass;
        }

        protected override Expr RewriteLeafCall(Call expr)
        {
            var outBuffer = expr.CheckedType switch
            {
                TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArrayExpr()),
                DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArrayExpr(), dt.AxisPolices, dt.Placement),
                _ => throw new ArgumentOutOfRangeException(nameof(expr), $"Unsupported type {expr.CheckedType}"),
            };
            return _selectionPass.SelectCall(expr, outBuffer);
        }
    }
}
