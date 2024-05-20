// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class BinaryEvaluator : ITypeInferencer<Binary>, IKernelInfoEvaluator<Binary>
{
    public IRType Visit(ITypeInferenceContext context, Binary target)
    {
        return TupleType.Void;
    }

    public MicroKernelInfo Visit(Binary op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions)
    {
        return new(Enumerable.Repeat(1, domain.Length).ToArray(), Enumerable.Repeat(new ValueRange<int>(1, int.MaxValue), domain.Length).ToArray(), 128, 128);
    }
}
