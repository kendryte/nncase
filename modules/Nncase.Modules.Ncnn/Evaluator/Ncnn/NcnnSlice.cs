// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnSlice"/>.
/// </summary>
public class NcnnSliceEvaluator : IEvaluator<NcnnSlice>, ITypeInferencer<NcnnSlice>, ICostEvaluator<NcnnSlice>, IShapeEvaluator<NcnnSlice>, IMetricEvaluator<NcnnSlice>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnSlice slice)
    {
        var input = context.GetOrtArgumentValue(slice, NcnnSlice.Input);
        var split = slice.Slices;
        var axis = slice.Axis;
        var result = OrtKI.Split(input, split, axis);
        return Value.FromTensors(result.Select(t => t.ToTensor()).ToArray());
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnSlice target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnSlice.Input);
        return Visit(input, target.Slices, target.Axis);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnSlice target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnSlice target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnSlice.Input);

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnSlice target) => context.GetArgumentShape(target, NcnnSlice.Input);

    private IRType Visit(TensorType input, int[] slices, int axis)
    {
        var outputInfo = new List<TensorType>();
        var outputShape = new List<List<int>>();
        for (int i = 0; i < slices.Length; i++)
        {
            outputShape.Add(input.Shape.ToValueList());
            outputShape[i][axis] = slices[i];
            outputInfo.Add(new TensorType(input.DType, outputShape[i].ToArray()));
        }

        return new TupleType(outputInfo);
    }
}
