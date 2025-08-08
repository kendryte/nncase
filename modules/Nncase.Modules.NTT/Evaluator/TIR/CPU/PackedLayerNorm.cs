// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Google.OrTools.ConstraintSolver;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PackedLayerNormEvaluator : ITypeInferencer<PackedLayerNorm>, IKernelInfoEvaluator<PackedLayerNorm>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, PackedLayerNorm target) => TupleType.Void;

    public MicroKernelInfo Visit(PackedLayerNorm op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Range(0, domain.Length).Select(i => 1).ToArray();
        var multipliers = Enumerable.Range(0, domain.Length).Select(i => i < op.Axis ? new ValueRange<long>(1, int.MaxValue) : new ValueRange<long>(context.BufferShapes[0][i], int.MaxValue)).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[2] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[3] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var factor = System.Math.Min(context.BufferShapes[0][0], 32);
        return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][0], solver.MakeIntConst(factor)));
    }
}
