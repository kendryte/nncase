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
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, (bufferShapes, solver) =>
        {
            var ashape = bufferShapes[0];
            var bshape = bufferShapes[1];
            var cshape = bufferShapes[2];
            var (k, m, n) = (ashape[^1], cshape[^2], cshape[^1]);
            var factor = 16 * 4 * 4;
            return factor * (1 + solver.MakeIsLessVar(k, solver.MakeIntConst(16)) + solver.MakeIsLessVar(n, solver.MakeIntConst(4)) + solver.MakeIsLessVar(m, solver.MakeIntConst(4)));
        });
    }
}
