// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Evaluator.NN;

/// <summary>
/// NN module.
/// </summary>
public class NNModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        // Activation
        builder.RegisterType<CeluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<EluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<HardSwishEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<LeakyReluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<PReluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SeluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SigmoidEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<HardSigmoidEvaluator>().AsImplementedInterfaces();

        // Convolution
        builder.RegisterType<Conv2DEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<Conv2DTransposeEvaluator>().AsImplementedInterfaces();

        // Normalization
        builder.RegisterType<BatchNormalizationEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<InstanceNormalizationEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<LRNEvaluator>().AsImplementedInterfaces();

        // ReduceWindow
        builder.RegisterType<ReduceWindow2DEvaluator>().AsImplementedInterfaces();

        // Soft*
        builder.RegisterType<LogSoftmaxEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SoftmaxEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SoftplusEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SoftsignEvaluator>().AsImplementedInterfaces();

        builder.RegisterType<BatchToSpaceEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<HardmaxEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<OneHotEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<PadEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SpaceToBatchEvaluator>().AsImplementedInterfaces();

        builder.RegisterType<CustomCallEvaluator>().AsImplementedInterfaces();
    }
}
