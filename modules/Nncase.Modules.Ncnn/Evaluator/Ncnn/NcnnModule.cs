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
    }
}
