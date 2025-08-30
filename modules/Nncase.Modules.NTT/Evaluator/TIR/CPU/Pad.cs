// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PadEvaluator : ITypeInferencer<Pad>, IKernelInfoEvaluator<Pad>
{
    public IRType Visit(ITypeInferenceContext context, Pad target)
    {
        context.CheckArgumentType<TensorType>(target, Pad.Input);
        context.CheckArgumentType<TensorType>(target, Pad.Output);
        return TupleType.Void;
    }

    public MicroKernelInfo Visit(Pad op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Range(0, domain.Length).Select(i => 1).ToArray();
        var multipliers = Enumerable.Range(0, domain.Length).Select(i => op.ActualPadAxes.Contains(i) ? new ValueRange<long>(1, int.MaxValue) : new ValueRange<long>(context.BufferShapes[0][i], int.MaxValue)).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][0], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][0], solver.MakeIntConst(factor)));
    }
}
