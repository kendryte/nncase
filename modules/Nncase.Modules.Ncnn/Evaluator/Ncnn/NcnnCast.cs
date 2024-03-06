// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnCast"/>.
/// </summary>
public class NcnnCastEvaluator : IEvaluator<NcnnCast>, ITypeInferencer<NcnnCast>, ICostEvaluator<NcnnCast>, IShapeEvaluator<NcnnCast>, IMetricEvaluator<NcnnCast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnCast cast)
    {
        var input = context.GetOrtArgumentValue(cast, NcnnCast.Input);
        return OrtKI.Cast(input, cast.ToType).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnCast target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnCast.Input);
        return Visit(input, target.ToType);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnCast target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnCast target)
    {
        _ = context.GetArgumentType<TensorType>(target, NcnnCast.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnCast target) => context.GetArgumentShape(target, NcnnCast.Input);

    public DataType RecoverDataType(int num)
    {
        return num switch
        {
            1 => DataTypes.Float32,
            2 => DataTypes.Float16,
            4 => DataTypes.BFloat16,
            _ => throw new DataException($"not support DataTypeNum :{num}"),
        };
    }

    private IRType Visit(TensorType input, int dT)
    {
        return new TensorType(RecoverDataType(dT), input.Shape);
    }
}
