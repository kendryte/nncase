// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Shapes;
using OrtKISharp;

namespace Nncase.Evaluator.Shapes;

/// <summary>
/// Evaluator for <see cref="AsTensor"/>.
/// </summary>
public class AsTensorEvaluator : IEvaluator<AsTensor>, ITypeInferencer<AsTensor>, ICostEvaluator<AsTensor>, IMetricEvaluator<AsTensor>
{
    public IValue Visit(IEvaluateContext context, AsTensor shape)
    {
        var input = context.GetArgumentValueAsTensor(shape, AsTensor.Input);
        return Value.FromTensor(input);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, AsTensor target)
    {
        var input = context.CheckArgumentType<IRType>(target, AsTensor.Input);
        return input switch
        {
            DimensionType => TensorType.Scalar(DataTypes.Int64),
            ShapeType st => new TensorType(DataTypes.Int64, [st.Rank ?? Dimension.Unknown]),
            _ => new InvalidType(input.GetType().Name),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, AsTensor target)
    {
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, AsTensor target)
    {
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType),
        };
    }
}
