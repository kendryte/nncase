// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.IR.NTT;

/// <summary>
/// CPU module.
/// </summary>
internal class NTTModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<LoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<VectorizedReduceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<VectorizedSoftMaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<VectorizedLayerNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<VectorizedMatMulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<VectorizedBinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Im2colEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<InstanceNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ResizeImageEvaluator>(reuse: Reuse.Singleton);
    }
}
