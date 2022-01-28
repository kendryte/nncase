// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Math module.
/// </summary>
public class NNModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<BinaryEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ClampEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<CompareEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<CumSumEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<MatMulEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReduceEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReduceArgEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<UnaryEvaluator>().AsImplementedInterfaces();
    }
}
