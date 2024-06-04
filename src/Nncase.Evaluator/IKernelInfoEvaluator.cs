// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;

namespace Nncase.Evaluator;

public interface IKernelInfoEvaluator
{
    MicroKernelInfo Visit(Op op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions);
}

public interface IKernelInfoEvaluator<T> : IKernelInfoEvaluator
    where T : Op
{
    MicroKernelInfo Visit(T op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions);

    MicroKernelInfo IKernelInfoEvaluator.Visit(Op op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions)
    {
        return Visit((T)op, domain, accessMaps, bufferShapes, targetOptions);
    }
}
