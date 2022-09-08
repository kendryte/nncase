// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Where"/>.
/// </summary>
public class WhereEvaluator : IEvaluator<Where>, ITypeInferencer<Where>, ICostEvaluator<Where>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Where where)
    {
        var xt = context.GetArgumentValueAsTensor(where, Where.X);
        var yt = context.GetArgumentValueAsTensor(where, Where.Y);
        if (xt.Dimensions[0] == 0 && yt.Dimensions[0] == 0 && xt.ElementType == DataTypes.Float32)
        {
            var condTensor = context.GetArgumentValueAsTensor<bool>(where, Where.Cond);
            if (condTensor.Rank > 1)
            {
                throw new NotImplementedException();
            }
            var result = condTensor.Select((b, i) => (b, i)).Where(t => t.b).Select(t => (long)t.i).ToArray();
            return Value.FromTensor(Tensor.FromSpan<long>(result, new Shape(result.Length, condTensor.Rank)));
        }
        var cond = context.GetOrtArgumentValue(where, Where.Cond);
        var x = context.GetOrtArgumentValue(where, Where.X);
        var y = context.GetOrtArgumentValue(where, Where.Y);
        return OrtKI.Where(cond, x, y).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Where target)
    {
        var cond = context.CheckArgumentType<TensorType>(target, Where.Cond);
        var x = context.CheckArgumentType<TensorType>(target, Where.X);
        var y = context.CheckArgumentType<TensorType>(target, Where.Y);
        if (IsTFWhere(x, y))
        {
            // dim[0] = count_nonzero(cond)
            return new TensorType(DataTypes.Int64, new Shape(Dimension.Unknown, cond.Shape.Rank));
        }
        return TypeInference.BroadcastType(x.DType, cond, x, y);
    }

    private bool IsTFWhere(TensorType x, TensorType y)
    {
        return x.Shape[0] == 0 && y.Shape[0] == 0 && x.DType == DataTypes.Float32;
    }

    public Cost? Visit(ICostEvaluateContext context, Where target)
    {
        var cond = context.GetArgumentType<TensorType>(target, Where.Cond);
        var x = context.GetArgumentType<TensorType>(target, Where.X);
        var y = context.GetArgumentType<TensorType>(target, Where.Y);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(cond, x, y),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(cond, CostUtility.GetCPUCyclesOfCompare()),
        };
    }
}
