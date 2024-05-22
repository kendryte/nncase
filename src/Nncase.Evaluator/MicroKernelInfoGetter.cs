// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;

namespace Nncase.Evaluator;

internal sealed class MicroKernelInfoProvider : IMicroKernelInfoProvider
{
    private readonly IServiceProvider _serviceProvider;

    public MicroKernelInfoProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public MicroKernelInfo GetInfo(Op op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions)
    {
        var evaluatorType = typeof(IKernelInfoEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (IKernelInfoEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(op, domain, accessMaps, bufferShapes, targetOptions);
    }
}
