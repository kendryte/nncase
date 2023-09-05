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

public partial class SqueezeShapeEvaluator : IEvaluator<SqueezeShape>, ITypeInferencer<SqueezeShape>, ICostEvaluator<SqueezeShape>, IShapeEvaluator<SqueezeShape>, IMetricEvaluator<SqueezeShape>
{
    public IValue Visit(IEvaluateContext context, SqueezeShape target)
    {
        var input = context.GetArgumentValueAsTensor(target, SqueezeShape.Input);
        var dims = context.GetArgumentValueAsTensor(target, SqueezeShape.Dim);
        var t = IR.F.Tensors.Squeeze(input, dims);
        return ShapeExprUtility.GetShapeValue(t);
    }

    public IRType Visit(ITypeInferenceContext context, SqueezeShape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, SqueezeShape.Input);
        var dims = context.CheckArgumentType<TensorType>(target, SqueezeShape.Dim);
        return new TensorType(DataTypes.Int64, new[] { input.Shape.Rank - dims.Shape[0] });
    }

    public Cost Visit(ICostEvaluateContext context, SqueezeShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, SqueezeShape target)
    {
        var input = context.GetArgument(target, SqueezeShape.Input);
        var dims = context.GetArgument(target, SqueezeShape.Dim);
        return new[] { input.CheckedShape.Rank - dims.CheckedShape[0] };
    }

    public Metric Visit(IMetricEvaluateContext context, SqueezeShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
