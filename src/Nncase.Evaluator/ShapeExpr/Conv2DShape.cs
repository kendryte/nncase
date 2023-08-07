// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.ShapeExpr;

namespace Nncase.Evaluator.ShapeExpr;

[EvaluatorGenerator]
[TypeInferGenerator]
public partial class Conv2DShapeEvaluator : IEvaluator<Conv2DShape>, ITypeInferencer<Conv2DShape>, ICostEvaluator<Conv2DShape>, IShapeEvaluator<Conv2DShape>, IMetricEvaluator<Conv2DShape>
{
    public IValue Visit(Tensor input, Tensor weights, Tensor padding, Tensor stride, Tensor dilation, Tensor groups)
    {
        var ty = TypeInference.Conv2DType(GetTensorType(input), GetTensorType(weights), stride, padding, dilation, groups);
        var shape = ty switch
        {
            TensorType tensorType => tensorType.Shape,
            _ => throw new InvalidOperationException(),
        };
        if (!shape.IsFixed)
        {
            throw new InvalidOperationException();
        }

        return Value.FromTensor(shape.Select(x => (long)x.FixedValue).ToArray());
    }

    public IRType Visit()
    {
        return new TensorType(DataTypes.Int64, new Shape(4));
    }

    public Cost Visit(ICostEvaluateContext context, Conv2DShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, Conv2DShape target)
    {
        return new[] { 4 };
    }

    public Metric Visit(IMetricEvaluateContext context, Conv2DShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    private TensorType GetTensorType(Tensor input)
    {
        return new TensorType(DataTypes.Float32, input.ToArray<int>());
    }
}
