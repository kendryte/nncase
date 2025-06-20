﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Imaging;
using Nncase.Evaluator.Math;
using Nncase.Evaluator.Tensors;
using Nncase.Evaluator.TIR.CPU;
using Nncase.Hosting;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.TIR.NTT;

/// <summary>
/// CPU module.
/// </summary>
internal class NTTModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BinaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GetItemEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatmulEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SUMMAEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PtrOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SramPtrEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TensorLoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TensorStoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnaryEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedSoftMaxEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedLayerNormEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PackedBinaryEvaluator>(reuse: Reuse.Singleton);
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
        registrator.RegisterManyInterface<ReduceArgEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherReduceScatterEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ClampEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CastEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<WhereEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ExpandEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ErfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CompareEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ScatterNDEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ShapeOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConstantOfShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<RangeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PagedAttentionEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UpdatePagedAttentionKVCacheEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CreatePagedAttentionKVCacheEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<IdentityPagedAttentionKVCacheEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherPagedAttentionKVCacheEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GetPositionIdsEvaluator>(reuse: Reuse.Singleton);
    }
}
