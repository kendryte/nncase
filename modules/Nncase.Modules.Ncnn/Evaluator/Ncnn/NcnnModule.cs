// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Ncnn module.
/// </summary>
internal class NcnnModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<NcnnSoftmaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnUnaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnBatchNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnBinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnCeluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnClipEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnConcatEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnConvEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnCumsumEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnEluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnErfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnHardSigmoidEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnHardSwishEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnInstanceNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnLRNEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnLSTMEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnPaddingEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnPoolingEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnPReluEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnReductionEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnReshapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnSELUEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnSigmoidEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnCropEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnSoftplusEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NcnnSliceEvaluator>(reuse: Reuse.Singleton);
    }
}
