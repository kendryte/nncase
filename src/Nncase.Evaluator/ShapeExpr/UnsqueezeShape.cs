// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.ShapeExpr;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Evaluator.ShapeExpr;

public partial class UnsqueezeShapeEvaluator : IEvaluator<UnsqueezeShape>, ITypeInferencer<UnsqueezeShape>, ICostEvaluator<UnsqueezeShape>, IShapeEvaluator<UnsqueezeShape>, IMetricEvaluator<UnsqueezeShape>
{
    public IValue Visit(IEvaluateContext context, UnsqueezeShape target)
    {
        var input = context.GetArgumentValueAsTensor(target, UnsqueezeShape.Input);
        var dims = context.GetArgumentValueAsTensor(target, UnsqueezeShape.Dim);
        var t = IR.F.Tensors.Unsqueeze(input, dims);
        return ShapeExprUtility.GetShapeValue(t);
    }

    public IRType Visit(ITypeInferenceContext context, UnsqueezeShape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, UnsqueezeShape.Input);
        var dims = context.CheckArgumentType<TensorType>(target, UnsqueezeShape.Dim);
        return new TensorType(DataTypes.Int64, new[] { input.Shape.Rank + dims.Shape[0] });
    }

    public Cost Visit(ICostEvaluateContext context, UnsqueezeShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, UnsqueezeShape target)
    {
        var input = context.GetArgument(target, UnsqueezeShape.Input);
        var dims = context.GetArgument(target, UnsqueezeShape.Dim);
        return new[] { input.CheckedShape.Rank + dims.CheckedShape[0] };
    }

    public Metric Visit(IMetricEvaluateContext context, UnsqueezeShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
