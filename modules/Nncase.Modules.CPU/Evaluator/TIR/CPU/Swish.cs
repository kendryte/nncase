﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class SwishEvaluator : ITypeInferencer<Swish>, IKernelInfoEvaluator<Swish>
{
    public IRType Visit(ITypeInferenceContext context, Swish target)
    {
        context.CheckArgumentType<TensorType>(target, Swish.Input);
        context.CheckArgumentType<TensorType>(target, Swish.Output);
        return TupleType.Void;
    }

    public MicroKernelInfo Visit(Swish swish, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Repeat(1, domain.Length).ToArray();
        var multipliers = Enumerable.Repeat(new ValueRange<int>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (ICpuTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
    }
}
