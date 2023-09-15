// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Tensors;
using Nncase.Hosting;
using Nncase.IR.ShapeExpr;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.ShapeExpr;

/// <summary>
/// ShapeExpr module.
/// </summary>
internal class ShapeExprModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BroadcastShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Conv2DShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Conv2DTransposeShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<MatMulShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<GetPaddingsEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ReshapeShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SqueezeShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UnsqueezeShapeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<TransposeShapeEvaluator>(reuse: Reuse.Singleton);
    }
}
