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

public partial class BroadcastShapeEvaluator : IEvaluator<BroadcastShape>, ITypeInferencer<BroadcastShape>, ICostEvaluator<BroadcastShape>, IShapeEvaluator<BroadcastShape>, IMetricEvaluator<BroadcastShape>
{
    public IValue Visit(IEvaluateContext context, BroadcastShape broadcastShape)
    {
        var inputs = context.GetArgumentValueAsTensors(broadcastShape, BroadcastShape.Inputs);
        var type = TypeInference.BroadcastType(inputs.Select(input => new TensorType(DataTypes.Int32, input.ToArray<int>())).ToArray());
        var shape = type switch
        {
            TensorType tt => tt.Shape.ToValueArray().Select(x => (long)x).ToArray(),
            InvalidType it => throw new InvalidOperationException(it.Reason),
            _ => throw new InvalidOperationException("Unknown IRType"),
        };

        return Value.FromTensor(shape);
    }

    public IRType Visit(ITypeInferenceContext context, BroadcastShape target)
    {
        // var inputs = context.CheckArgumentType<TupleType>(target, BroadcastShape.Inputs);
        // var field = inputs.Fields.ToArray().MaxBy(ty => ty switch
        // {
        // TensorType tensorType => tensorType.Shape.Rank,
        // _ => throw new ArgumentOutOfRangeException(nameof(ty)),
        // })!;
        return new TensorType(DataTypes.Int64, new[] { Dimension.Unknown });
    }

    public Cost Visit(ICostEvaluateContext context, BroadcastShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, BroadcastShape target)
    {
        var inShape = context.GetArgumentShape(target, BroadcastShape.Inputs);

        // var ranks = ((IR.Tuple)inShape).Fields.ToArray().Select(shape =>
        // {
        //     return IR.F.Tensors.Cast(shape[0], DataTypes.Int32);
        // }).ToArray();
        var len = ((IR.Tuple)inShape).Fields.ToArray().Aggregate((Expr)1, (i, call) => IR.F.Math.Max(i, call));
        var bn = IR.F.Tensors.Cast(len, DataTypes.Int32);

        // DumpScope.Current.DumpIR(bn, "binary");
        return bn;
    }

    public Metric Visit(IMetricEvaluateContext context, BroadcastShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
