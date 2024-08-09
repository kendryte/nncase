// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Imaging;
using Nncase.Evaluator.NN;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;

namespace Nncase.Evaluator.TIR.CPU;

/// <summary>
/// CPU module.
/// </summary>
internal class CPUModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatmulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MemcopyEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PtrOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SramPtrEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TensorLoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TensorStoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedSoftMaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedLayerNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedMatMulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedBinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedTransposeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnpackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SliceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConcatEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SwishEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TransposeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReshapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Im2colEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<InstanceNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ResizeImageEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Conv2DEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReduceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherReduceScatterEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ClampEvaluator>(reuse: Reuse.Singleton);
    }
}
