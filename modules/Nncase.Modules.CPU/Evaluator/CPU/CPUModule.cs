// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.IR.CPU;

/// <summary>
/// CPU module.
/// </summary>
internal class CPUModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BoxingEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CPUKernelOpEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<LoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedSoftMaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedLayerNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedBinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedTransposeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnpackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Im2colEvaluator>(reuse: Reuse.Singleton);
    }
}
