// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using OrtKISharp;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.IR.CPU;

/// <summary>
/// Evaluator for <see cref="ResizeImage"/>.
/// </summary>
public class ResizeImageEvaluator : IEvaluator<ResizeImage>, ITypeInferencer<ResizeImage>, ICostEvaluator<ResizeImage>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ResizeImage target)
    {
        return OnnxResize(context, target);
    }

    public IValue OnnxResize(IEvaluateContext context, ResizeImage target)
    {
        var input = context.GetOrtArgumentValue(target, ResizeImage.Input);
        var sizes = target.NewSize;

        input = CPUEvaluatorUtility.UnpackTensor(input, target.PackedAxes, target.PadedNums, out var lanes);
        OrtKISharp.Tensor resized = OrtKI.ResizeWithSizes(
           input,
           Array.Empty<float>(),
           sizes.Select(i => (long)i).ToArray(),
           ResizeModeHelper.ToString(target.TransformationMode),
           -0.75f,
           0,
           0f,
           ResizeModeHelper.ToString(target.ResizeMode),
           ResizeModeHelper.ToString(target.NearestMode));

        resized = CPUEvaluatorUtility.RepackTensor(resized, lanes, target.PackedAxes, target.PadedNums);
        if (lanes.Count > 0)
        {
            return Value.FromTensor(Tensor.FromBytes(new TensorType(new VectorType(DataTypes.Float32, lanes), resized.Shape.Take(4).Select(i => (int)i).ToArray()), resized.BytesBuffer.ToArray()));
        }

        return resized.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ResizeImage target)
    {
        var input = context.CheckArgumentType<IRType>(target, ResizeImage.Input);
        var newSize = target.NewSize.ToArray();

        return input switch
        {
            TensorType t => Visit(t, newSize),
            DistributedType d => Visit(d, newSize),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    public IRType Visit(TensorType input, Expr newSize)
    {
        return TypeInference.ResizeType(input, newSize, null);
    }

    public IRType Visit(DistributedType input, Expr newSize)
    {
        if (Visit(input.TensorType, newSize) is not TensorType tensorType)
        {
            return new InvalidType(string.Empty);
        }

        var ndsbp = new SBP[input.Placement.Rank];

        var invalid = new InvalidType($"{input}, not support");
        for (int i = 0; i < input.Placement.Rank; i++)
        {
            switch (input.NdSBP[i])
            {
                case SBPSplit { Axis: int ix } when ix < 2:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case SBPBroadCast:
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ResizeImage target)
    {
        var inputType = context.GetArgumentType<IRType>(target, ResizeImage.Input);
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, 4),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, ResizeImage target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ResizeImage.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(returnType) * MetricUtility.ResizeLinearFLOPs,
        };
    }
}
