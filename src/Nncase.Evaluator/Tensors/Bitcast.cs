// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Bitcast"/>.
/// </summary>
public class BitcastEvaluator : IEvaluator<Bitcast>, ITypeInferencer<Bitcast>, ICostEvaluator<Bitcast>, IMetricEvaluator<Bitcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Bitcast cast)
    {
        var input = context.GetArgumentValue(cast, Bitcast.Input).AsTensor();
        return Value.FromTensor(input.CastTo(cast.NewType, CastMode.Reinterpret));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var input = context.CheckArgumentType<IRType>(target, Bitcast.Input);
        return input switch
        {
            TensorType t => Visit(target, t),
            DistributedType d => Visit(target, d),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Bitcast target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Bitcast target)
    {
        return new();
    }

    private IRType Visit(Bitcast target, TensorType input)
    {
        if (input.Shape is not RankedShape rankedInShape)
        {
            return new InvalidType(input.ToString());
        }

        var srcSize = input.DType.SizeInBytes;
        var destSize = target.NewType.SizeInBytes;
        var newDimensions = rankedInShape.Dimensions.ToArray();

        if (srcSize != destSize)
        {
            if (newDimensions.Rank == 0)
            {
                newDimensions = [srcSize / destSize];
            }
            else
            {
                newDimensions[^1] = newDimensions[^1] * srcSize / destSize;
            }
        }

        return new TensorType(target.NewType, newDimensions);
    }

    private IRType Visit(Bitcast target, DistributedType input)
    {
        var tensorType = Visit(target, input.TensorType);
        if (tensorType is not TensorType outTensorType)
        {
            return tensorType;
        }

        var invalid = new InvalidType(input.ToString());
        var ndsbp = new SBP[input.AxisPolicies.Count];
        for (int i = 0; i < ndsbp.Length; i++)
        {
            if (input.AxisPolicies[i] is SBPPartial)
            {
                return invalid;
            }

            ndsbp[i] = input.AxisPolicies[i];
        }

        return new DistributedType(outTensorType, ndsbp, input.Placement);
    }
}
