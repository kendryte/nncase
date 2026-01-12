// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.NN;

public class GridSampleEvaluator : IEvaluator<GridSample>, ITypeInferencer<GridSample>, ICostEvaluator<GridSample>, IMetricEvaluator<GridSample>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, GridSample target)
    {
        var input = context.GetOrtArgumentValue(target, GridSample.Input);
        var grid = context.GetOrtArgumentValue(target, GridSample.Grid);

        return OrtKI.GridSample(
            input,
            grid,
            GridSampleModeHelper.ToInt(target.AlignCorners),
            GridSampleModeHelper.ToString(target.Mode),
            GridSampleModeHelper.ToString(target.PaddingMode)).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, GridSample target)
    {
        var input = context.CheckArgumentType<IRType>(target, GridSample.Input);
        var grid = context.CheckArgumentType<IRType>(target, GridSample.Grid);

        return input switch
        {
            TensorType t => Visit(t, (TensorType)grid),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    public IRType Visit(TensorType input, TensorType grid)
    {
        var newShape = input.Shape.ToArray()[..2].Concat(grid.Shape.ToArray()[1..^1]).ToArray();
        return input with { Shape = new Shape(newShape) };
    }

    public Cost Visit(ICostEvaluateContext context, GridSample target)
    {
        var inputType = context.GetArgumentType<IRType>(target, GridSample.Input);
        var gridType = context.GetArgumentType<IRType>(target, GridSample.Grid);
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(gridType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, 4),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, GridSample target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, GridSample.Input);
        var gridType = context.GetArgumentType<TensorType>(target, GridSample.Grid);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(gridType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(returnType) * MetricUtility.ResizeLinearFLOPs,
        };
    }
}
