// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.ArgsStruct;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnCrop"/>.
/// </summary>
public class NcnnCropEvaluator : IEvaluator<NcnnCrop>, ITypeInferencer<NcnnCrop>, ICostEvaluator<NcnnCrop>, IShapeEvaluator<NcnnCrop>, IMetricEvaluator<NcnnCrop>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnCrop crop)
    {
        var input = context.GetOrtArgumentValue(crop, NcnnCrop.Input);
        var starts = crop.Args.Starts;
        var ends = crop.Args.Ends;
        var axes = crop.Args.Axes;
        var steps = Enumerable.Repeat(1, (starts ?? Array.Empty<int>()).Length).ToArray();
        return OrtKI.Slice(input, starts ?? Array.Empty<int>(), ends ?? Array.Empty<int>(), axes ?? Array.Empty<int>(), steps).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnCrop target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnCrop.Input);
        return Visit(input, target.Args);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnCrop target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnCrop target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnCrop.Input);

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnCrop target) => context.GetArgumentShape(target, NcnnCrop.Input);

    private IRType Visit(TensorType input, CropArgs args)
    {
        var outputShape = input.Shape.ToArray();
        for (int i = 0; i < args.Axes.Length; i++)
        {
            var tStart = args.Starts[i] >= 0 ? args.Starts[i] : args.Starts[i] + outputShape[args.Axes[i]].FixedValue;
            var tEnd = args.Ends[i] >= 0 ? args.Ends[i] : args.Ends[i] + outputShape[args.Axes[i]].FixedValue;
            outputShape[args.Axes[i] < 0 ? args.Axes[i] + outputShape.Length : args.Axes[i]] = System.Math.Abs(tEnd - tStart);
        }

        return new TensorType(input.DType, outputShape);
    }
}
