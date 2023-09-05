// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.ShapeExpr;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Evaluator.ShapeExpr;

public partial class TransposeShapeEvaluator : IEvaluator<TransposeShape>, ITypeInferencer<TransposeShape>, ICostEvaluator<TransposeShape>, IShapeEvaluator<TransposeShape>, IMetricEvaluator<TransposeShape>
{
    public IValue Visit(IEvaluateContext context, TransposeShape target)
    {
        var input = context.GetArgumentValueAsTensor(target, TransposeShape.Input);
        var perm = context.GetArgumentValueAsTensor(target, TransposeShape.Perm);
        var t = IR.F.Tensors.Transpose(input, perm);
        return ShapeExprUtility.GetShapeValue(t);
    }

    public IRType Visit(ITypeInferenceContext context, TransposeShape target)
    {
        var tt = context.CheckArgumentType<TensorType>(target, TransposeShape.Input);
        return new TensorType(DataTypes.Int64, new[] { tt.Shape.Rank });
    }

    public Cost Visit(ICostEvaluateContext context, TransposeShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, TransposeShape target)
    {
        var input = context.GetArgument(target, TransposeShape.Input);
        if (input.CheckedShape.IsUnranked)
        {
            return IR.F.Tensors.Rank(input);
        }
        return input.CheckedShape.Rank;
    }

    public Metric Visit(IMetricEvaluateContext context, TransposeShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
