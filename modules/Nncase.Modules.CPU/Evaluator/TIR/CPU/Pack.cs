// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class PackEvaluator : ITypeInferencer<Pack>, IKernelInfoEvaluator<Pack>
{
    public IRType Visit(ITypeInferenceContext context, Pack target) => TupleType.Void;

    public MicroKernelInfo Visit(Pack op, MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        var primitives = Enumerable.Repeat(1, domain.Length).ToArray();
        var multipliers = Enumerable.Repeat(new ValueRange<int>(1, int.MaxValue), domain.Length).ToArray();
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        var opt = (ICpuTargetOptions)context.TargetOptions;
        bufferInfos[0] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Read);
        bufferInfos[1] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(primitives, multipliers, bufferInfos, (bufferShapes, solver) =>
        {
            var ashape = bufferShapes[0];
            var bshape = bufferShapes[1];
            var factor = System.Math.Min(context.BufferShapes[0][^1], 32);
            return factor * (1 + solver.MakeIsLessVar(bufferShapes[0][^1], solver.MakeIntConst(factor)));
        });
    }
}
