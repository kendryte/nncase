// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnPermute"/>.
/// </summary>
public class NcnnPermuteEvaluator : IEvaluator<NcnnPermute>, ITypeInferencer<NcnnPermute>, ICostEvaluator<NcnnPermute>, IShapeEvaluator<NcnnPermute>, IMetricEvaluator<NcnnPermute>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnPermute permute)
    {
        var input = context.GetOrtArgumentValue(permute, NcnnPermute.Input);
        return OrtKI.Transpose(input, permute.Perm.ToLongs()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnPermute target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnPermute.Input);
        return Visit(input, target.Perm);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnPermute target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnPermute target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnPermute.Input);

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnPermute target) => context.GetArgumentShape(target, NcnnPermute.Input);

    private IRType Visit(TensorType input, int[] perm)
    {
        var outputShape = input.Shape.ToValueList();
        if (perm.Length - input.Shape.Count == 1)
        {
            perm = perm.Remove(0);
            var realPerm = perm.Select(x => x - 1).ToArray();
            for (int i = 0; i < realPerm.Length; i++)
            {
                outputShape[i] = input.Shape.ToValueList()[realPerm[i]];
            }
        }

        return new TensorType(input.DType, outputShape.ToArray());
    }
}
