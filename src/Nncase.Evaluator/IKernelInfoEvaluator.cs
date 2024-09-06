// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;

namespace Nncase.Evaluator;

public interface IKernelInfoEvaluator
{
    MicroKernelInfo Visit(Op op, MicroKernelContext context);
}

public interface IKernelInfoEvaluator<T> : IKernelInfoEvaluator
    where T : Op
{
    MicroKernelInfo Visit(T op, MicroKernelContext context);

    MicroKernelInfo IKernelInfoEvaluator.Visit(Op op, MicroKernelContext context)
    {
        return Visit((T)op, context);
    }
}
