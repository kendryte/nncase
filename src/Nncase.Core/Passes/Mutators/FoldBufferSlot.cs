// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// remove buffer BaseMentOf/DDrOf/MmuOF.
/// </summary>
public sealed class FoldBufferSlot : ExprRewriter
{
    protected internal override Expr VisitPrimFunction(TIR.PrimFunction expr, Unit context)
    {
        if (expr.SchedResult.IsScheduled == true)
        {
            return base.VisitPrimFunction(expr, context);
        }

        return expr;
    }

    protected override Expr RewriteLeafCall(Call expr)
    {
        if (expr.Target is IR.Buffers.BaseMentOf)
        {
            var locate = ((TIR.PhysicalBuffer)expr.Arguments[0]).MemLocation;
            return locate switch
            {
                MemoryLocation.Input => 0,
                MemoryLocation.Output => 1,
                MemoryLocation.Rdata => 2,
                MemoryLocation.Data => 3,
                _ => throw new ArgumentOutOfRangeException($"You Can't Assgin The BaseMent For {locate}!"),
            };
        }
        else if (expr.Target is IR.Buffers.DDrOf)
        {
            if (expr.Arguments[0] is TIR.PhysicalBuffer buf)
            {
                return buf.Start;
            }
        }

        return expr;
    }
}
