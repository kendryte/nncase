// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Math module.
/// </summary>
public class MathModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<BinaryEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ClampEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<CompareEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<CumSumEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<DequantizeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<FakeDequantizeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<FakeQuantizeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<MatMulEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<QuantizeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<QuantParamOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<RangeOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReduceEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReduceArgEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<UnaryEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SelectEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<RequireEvaluator>().AsImplementedInterfaces();
    }
}
