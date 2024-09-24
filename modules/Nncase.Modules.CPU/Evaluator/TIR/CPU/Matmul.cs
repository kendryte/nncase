// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class MatmulEvaluator : ITypeInferencer<Matmul>, IKernelInfoEvaluator<Matmul>
{
    public IRType Visit(ITypeInferenceContext context, Matmul target) => TupleType.Void;

    public MicroKernelInfo Visit(Matmul op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Repeat(1, domain.Length).ToArray();
        var multipliers = Enumerable.Repeat(new ValueRange<int>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (ICpuTargetOptions)context.TargetOptions;
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
        var kb = context.BufferShapes[0][^1];

        // var (kb, mb, nb) = (context.BufferShapes[0][^1], context.BufferShapes[^1][^2], context.BufferShapes[^2][^1]);
        // note add constrants in here will cause solver fail.
        // if (kb % 8 == 0 && mb % 8 == 0 && nb % 8 == 0)
        // {
        // solver.Add(solver.MakeEquality(k, 8));
        // solver.Add(solver.MakeEquality(m, 8));
        // solver.Add(solver.MakeEquality(n, 8));
        // return solver.MakeIntConst(1);
        // }
        return (kb - k) * m * n * 2;
    }
}
