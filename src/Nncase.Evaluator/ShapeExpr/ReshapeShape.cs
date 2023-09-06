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

public partial class ReshapeShapeEvaluator : IEvaluator<ReshapeShape>, ITypeInferencer<ReshapeShape>, ICostEvaluator<ReshapeShape>, IShapeEvaluator<ReshapeShape>, IMetricEvaluator<ReshapeShape>
{
    public IValue Visit(IEvaluateContext context, ReshapeShape target)
    {
        var inShape = context.GetArgumentValueAsArray<int>(target, ReshapeShape.InputShape);
        var shape = context.GetArgumentValueAsTensor(target, ReshapeShape.Shape);
        var t = IR.F.Tensors.Reshape(new Var(new TensorType(DataTypes.Float32, inShape)), shape);
        return ShapeExprUtility.GetShapeValue(t);
    }

    public IRType Visit(ITypeInferenceContext context, ReshapeShape target)
    {
        var shape = context.CheckArgumentType<TensorType>(target, ReshapeShape.Shape);
        return new TensorType(DataTypes.Int64, shape.Shape.ToValueArray());
    }

    public Cost Visit(ICostEvaluateContext context, ReshapeShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, ReshapeShape target)
    {
        var shape = context.GetArgument(target, ReshapeShape.Shape);
        return shape.CheckedShape.ToValueArray();
    }

    public Metric Visit(IMetricEvaluateContext context, ReshapeShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
