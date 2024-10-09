// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Google.OrTools.ConstraintSolver;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class PackedBinaryEvaluator : ITypeInferencer<PackedBinary>, IKernelInfoEvaluator<PackedBinary>
{
    public IRType Visit(ITypeInferenceContext context, PackedBinary target) => TupleType.Void;

    public MicroKernelInfo Visit(PackedBinary op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Repeat(1, domain.Length).ToArray();
        var multipliers = Enumerable.Repeat(new ValueRange<int>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (ICpuTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[2] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, GetComputeCycle);
    }

    private static IntExpr GetComputeCycle(IntExpr[][] bufferShapes, Solver solver, MicroKernelContext context)
    {
        var factora = System.Math.Min(context.BufferShapes[0][^1], 32);
        var factorb = System.Math.Min(context.BufferShapes[1][^1], 32);
        return factora * factorb * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factora)) + solver.MakeIsLessVar(bufferShapes[1][^1], solver.MakeIntConst(factorb)));
    }
}
