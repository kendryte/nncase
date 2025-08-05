// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class PackedMatMulEvaluator : ITypeInferencer<PackedMatMul>, IKernelInfoEvaluator<PackedMatMul>
{
    public IRType Visit(ITypeInferenceContext context, PackedMatMul target) => TupleType.Void;

    public MicroKernelInfo Visit(PackedMatMul op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Repeat(1, domain.Length).ToArray();
        var multipliers = Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();

        var (k, m, n) = (context.BufferShapes[0][^1], context.BufferShapes[2][^2], context.BufferShapes[2][^1]);
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (INTTTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[2] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read | MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, GetComputeCycle);
    }

    private static Google.OrTools.ConstraintSolver.IntExpr GetComputeCycle(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, MicroKernelContext context)
    {
        var ashape = bufferShapes[0];
        var cshape = bufferShapes[2];
        var (k, m, n) = (ashape[^1], cshape[^2], cshape[^1]);

        // var kb = context.BufferShapes[0][^1];
        return 16000 * (1 + solver.MakeIsLessVar(k, solver.MakeIntConst(8)) + solver.MakeIsLessVar(n, solver.MakeIntConst(8)) + solver.MakeIsLessVar(m, solver.MakeIntConst(8)));
    }
}
