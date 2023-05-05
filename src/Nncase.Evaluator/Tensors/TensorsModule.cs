// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Tensors module.
/// </summary>
internal class TensorsModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BroadcastEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BitcastEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<CastEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConcatEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ConstantOfShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ExpandEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FlattenEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GatherNDEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ProdEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<RangeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReshapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReverseSequenceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ShapeOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SizeOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SliceEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SplitEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SqueezeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StackEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TileEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TopKEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TransposeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnsqueezeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<WhereEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GetItemEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<IndexOfEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FixShapeEvaluator>(reuse: Reuse.Singleton);
    }
}
