// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.NN;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;

namespace Nncase.Evaluator.XPU;

/// <summary>
/// XPU module.
/// </summary>
internal class XPUModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<TDMALoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TDMAStoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatmulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<LayerNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConcatEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SliceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SoftmaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TransposeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherReduceScatterEvaluator>(reuse: Reuse.Singleton);
    }
}
