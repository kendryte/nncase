// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.ShapeExpr;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.ShapeExpr;

[EvaluatorGenerator]
[TypeInferGenerator]
public partial class Conv2DTransposeShapeEvaluator : IEvaluator<Conv2DTransposeShape>, ITypeInferencer<Conv2DTransposeShape>, ICostEvaluator<Conv2DTransposeShape>, IShapeEvaluator<Conv2DTransposeShape>, IMetricEvaluator<Conv2DTransposeShape>
{
    public IValue Visit(Tensor input, Tensor weights, Tensor stride, Tensor dilation, Tensor padding, Tensor outputPadding, int groups)
    {
        var outShape = Util.GetConvTransposeOutputShape(input, weights, stride, outputPadding, padding, dilation, string.Empty, Cast(groups, DataTypes.Int64));
        return Cast(outShape, DataTypes.Int64).Evaluate();
    }

    public IRType Visit()
    {
        return new TensorType(DataTypes.Int64, new Shape(4));
    }

    public Cost Visit(ICostEvaluateContext context, Conv2DTransposeShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, Conv2DTransposeShape target)
    {
        return new[] { 4 };
    }

    public Metric Visit(IMetricEvaluateContext context, Conv2DTransposeShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
