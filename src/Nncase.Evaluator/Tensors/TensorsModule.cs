// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.Evaluator.Tensors;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Tensors module.
/// </summary>
public class TensorsModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<BroadcastEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<CastEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ConcatEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ConstantOfShapeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ExpandEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<FlattenEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<GatherEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<GatherNDEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ProdEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<RangeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReshapeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ReverseSequenceEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<ShapeOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SizeOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SliceEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SplitEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<SqueezeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<StackEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<TileEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<TransposeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<UnsqueezeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<WhereEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<GetItemEvaluator>().AsImplementedInterfaces();
    }
}
