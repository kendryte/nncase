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
        registrator.RegisterManyInterface<CeluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<EluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<HardSwishEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SwishEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<LeakyReluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PReluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Relu6Evaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SeluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SigmoidEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<HardSigmoidEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ErfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GeluEvaluator>(reuse: Reuse.Singleton);

        // Convolution
        registrator.RegisterManyInterface<Conv2DEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Conv2DTransposeEvaluator>(reuse: Reuse.Singleton);

        // Normalization
        registrator.RegisterManyInterface<L2NormalizationEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<LayerNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BatchNormalizationEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<InstanceNormalizationEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<LpNormalizationEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<LRNEvaluator>(reuse: Reuse.Singleton);

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
