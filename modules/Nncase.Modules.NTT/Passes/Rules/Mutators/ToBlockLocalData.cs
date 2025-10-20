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
/// Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional Load/Store.
/// </summary>
public sealed class ToBlockLocalData : ExprRewriter
{
    private readonly Dictionary<BaseExpr, TIR.PhysicalBuffer> _bufferMemo = new(ReferenceEqualityComparer.Instance);

    protected override BaseExpr VisitSequential(Sequential expr, Unit context)
    {
        for (int i = 0; i < expr.Fields.Length - 1; i++)
        {
            {
                if (expr.Fields[i] is Call { Target: TIR.NTT.VectorizedLayerNorm ln } lnCall
                    && expr.Fields[i + 1] is Call { Target: TIR.NTT.Matmul mm } mmCall
                    && ln.CSourcePath is not null && mm.CSourcePath is not null)
                {
                    var lnOutput = lnCall.Arguments[TIR.NTT.VectorizedLayerNorm.Output.Index];
                    var mmLhs = mmCall.Arguments[TIR.NTT.Matmul.Lhs.Index];
                    if (lnOutput is TIR.Buffer a && mmLhs is TIR.Buffer b && a.MemSpan.Buffer == b.MemSpan.Buffer && b.MemSpan.Buffer.Location == MemoryLocation.Data)
                    {
                        var newPhysicalBuffer = a.MemSpan.Buffer.With(location: MemoryLocation.BlockLocalData);
                        _bufferMemo.TryAdd(lnOutput, newPhysicalBuffer);
                        _bufferMemo.TryAdd(mmLhs, newPhysicalBuffer);
                    }
                }
            }

            {
                if (expr.Fields[i] is Call { Target: TIR.NTT.Matmul mm } mmCall
                    && expr.Fields[i + 1] is Call { Target: TIR.NTT.VectorizedLayerNorm ln } lnCall
                    && ln.CSourcePath is not null && mm.CSourcePath is not null)
                {
                    var lnInput = lnCall.Arguments[TIR.NTT.VectorizedLayerNorm.Input.Index];
                    var mmOutput = mmCall.Arguments[TIR.NTT.Matmul.Output.Index];
                    if (lnInput is TIR.Buffer a && mmOutput is TIR.Buffer b && a.MemSpan.Buffer == b.MemSpan.Buffer && b.MemSpan.Buffer.Location == MemoryLocation.Data)
                    {
                        var newPhysicalBuffer = a.MemSpan.Buffer.With(location: MemoryLocation.BlockLocalData);
                        _bufferMemo.TryAdd(lnInput, newPhysicalBuffer);
                        _bufferMemo.TryAdd(mmOutput, newPhysicalBuffer);
                    }
                }
            }

            {
                if (expr.Fields[i] is Call { Target: TIR.NTT.Matmul mm } mmCall
                    && expr.Fields[i + 1] is Call { Target: TIR.NTT.SynchronizeThreads }
                    && expr.Fields[i + 2] is Call { Target: TIR.NTT.UpdatePagedAttentionKVCache }
                    && expr.Fields[i + 3] is Call { Target: TIR.NTT.UpdatePagedAttentionKVCache } upkvCall2
                    && mm.CSourcePath is not null)
                {
                    var mmOutput = mmCall.Arguments[TIR.NTT.Matmul.Output.Index];
                    var upkvInput = upkvCall2.Arguments[TIR.NTT.UpdatePagedAttentionKVCache.Slots.Index];
                    if (upkvInput is TIR.Buffer a && mmOutput is TIR.Buffer b && a.MemSpan.Buffer == b.MemSpan.Buffer && b.MemSpan.Buffer.Location == MemoryLocation.Data)
                    {
                        var newPhysicalBuffer = a.MemSpan.Buffer.With(location: MemoryLocation.BlockLocalData);
                        _bufferMemo.TryAdd(upkvInput, newPhysicalBuffer);
                        _bufferMemo.TryAdd(mmOutput, newPhysicalBuffer);
                    }
                }
            }

            // FIXME: need to add sync after primfunc.
            // {
            //     if (expr.Fields[i] is Call { Target: TIR.NTT.Matmul mm } mmCall
            //         && expr.Fields[i + 1] is Call { Target: TIR.PrimFunction } fnCall
            //         && mm.CSourcePath is not null)
            //     {
            //         var mmOutput = mmCall.Arguments[TIR.NTT.Matmul.Output.Index];
            //         foreach (var arg in fnCall.Arguments)
            //         {
            //             if (arg is TIR.Buffer a && mmOutput is TIR.Buffer b && a.MemSpan.Buffer == b.MemSpan.Buffer && b.MemSpan.Buffer.Location == MemoryLocation.Data)
            //             {
            //                 var newPhysicalBuffer = a.MemSpan.Buffer.With(location: MemoryLocation.BlockLocalData);
            //                 _bufferMemo.TryAdd(arg, newPhysicalBuffer);
            //                 _bufferMemo.TryAdd(mmOutput, newPhysicalBuffer);
            //             }
            //         }
            //     }
            // }
        }

        return base.VisitSequential(expr, context);
    }

    protected override BaseExpr RewriteLeafBuffer(TIR.Buffer expr)
    {
        if (_bufferMemo.TryGetValue(expr, out var buffer))
        {
            return expr.With(memSpan: expr.MemSpan.With(buffer: buffer));
        }

        return expr;
    }
}
