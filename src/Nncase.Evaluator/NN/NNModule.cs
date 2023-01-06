// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.NN;

/// <summary>
/// NN module.
/// </summary>
internal class NNModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        // Activation
        builder.RegisterType<CeluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<EluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<HardSwishEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<LeakyReluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<PReluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<Relu6Evaluator>().AsImplementedInterfaces();
        builder.RegisterType<SeluEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SigmoidEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<HardSigmoidEvaluator>().AsImplementedInterfaces();

        // Convolution
        registrator.RegisterManyInterface<Conv2DEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Conv2DTransposeEvaluator>(reuse: Reuse.Singleton);

        // Normalization
        builder.RegisterType<L2NormalizationEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<BatchNormalizationEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<InstanceNormalizationEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<LpNormalizationEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<LRNEvaluator>().AsImplementedInterfaces();

        // ReduceWindow
        registrator.RegisterManyInterface<ReduceWindow2DEvaluator>(reuse: Reuse.Singleton);

        // Soft*
        registrator.RegisterManyInterface<LogSoftmaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SoftmaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SoftplusEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SoftsignEvaluator>(reuse: Reuse.Singleton);

        registrator.RegisterManyInterface<BatchToSpaceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<HardmaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<OneHotEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SpaceToBatchEvaluator>(reuse: Reuse.Singleton);
    }
}
