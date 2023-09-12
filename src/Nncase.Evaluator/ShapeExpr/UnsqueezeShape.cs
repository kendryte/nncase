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
        var inShape = context.GetArgumentValueAsArray<int>(target, UnsqueezeShape.InputShape);
        var dims = context.GetArgumentValueAsTensor(target, UnsqueezeShape.Dim);
        var t = IR.F.Tensors.Unsqueeze(new Var(new TensorType(DataTypes.Float32, inShape)), dims);
        return ShapeExprUtility.GetShapeValue(t);
    }

    public IRType Visit(ITypeInferenceContext context, UnsqueezeShape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, UnsqueezeShape.InputShape);
        var dims = context.CheckArgumentType<TensorType>(target, UnsqueezeShape.Dim);
        if (!input.Shape.IsFixed)
        {
            return new TensorType(DataTypes.Int64, new[] { Dimension.Unknown });
        }

        return new TensorType(DataTypes.Int64, new[] { input.Shape.Size + dims.Shape[0] });
    }

    public Cost Visit(ICostEvaluateContext context, UnsqueezeShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, UnsqueezeShape target)
    {
        var input = context.GetArgument(target, UnsqueezeShape.InputShape);
        var dims = context.GetArgument(target, UnsqueezeShape.Dim);
        return IR.F.Tensors.Stack(new IR.Tuple(new[] { IR.F.Tensors.ShapeOf(input)[0] + (long)dims.CheckedShape[0].FixedValue }), 0);
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
