// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Conv2DTranspose"/>.
/// </summary>
public class Conv2DTransposeEvaluator : IEvaluator<Conv2DTranspose>, ITypeInferencer<Conv2DTranspose>, ICostEvaluator<Conv2DTranspose>, IShapeEvaluator<Conv2DTranspose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Conv2DTranspose conv)
    {
        var input = context.GetOrtArgumentValue(conv, Conv2DTranspose.Input);
        var weights = context.GetOrtArgumentValue(conv, Conv2DTranspose.Weights);
        var bias = context.GetOrtArgumentValue(conv, Conv2DTranspose.Bias);
        var stride = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Stride);
        var outputShape = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.OutputShape);

        // [w:[left right] h:[top bottom]]
        var pads = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Padding);
        var outputPaddings = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.OutputPadding);
        var dilation = context.GetArgumentValueAsArray<long>(conv, Conv2DTranspose.Dilation);
        var groups = context.GetArgumentValueAsScalar<long>(conv, Conv2DTranspose.Groups);
        var kernelShape = weights.Shape;
        return OrtKI.ConvTranspose(
            input,
            OrtKI.Transpose(weights, new long[] { 1, 0, 2, 3 }),
            bias,
            "NOTSET",
            dilation,
            groups,
            new long[] { kernelShape[2], kernelShape[3] },
            outputPaddings,
            outputShape,
            pads,
            stride).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2DTranspose target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Conv2DTranspose.Input);
        if (context.GetArgument(target, Conv2DTranspose.OutputShape) is TensorConst outShapeValue)
        {
            return new TensorType(input.DType, new Shape(outShapeValue.Value.Cast<int>()));
        }
        else
        {
            return input with { Shape = Shape.Unknown(4) };
        }
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Conv2DTranspose target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Input);
        var weightsType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Weights);
        var biasType = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Bias);
        var weightsShape = context.GetArgumentType<TensorType>(target, Conv2DTranspose.Weights).Shape;
        var outputType = context.GetReturnType<TensorType>();

        var macPerElement = weightsShape[1] * weightsShape[2] * weightsShape[3];
        return new() { [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType), [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType), [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, (uint)macPerElement.FixedValue * 2), };
    }

    public Expr Visit(IShapeEvaluateContext context, Conv2DTranspose target)
    {
        var input = context.GetArgumentShape(target, Conv2DTranspose.Input);
        var weights = context.GetArgumentShape(target, Conv2DTranspose.Weights);
        var stride = context.GetArgument(target, Conv2DTranspose.Stride);
        var dilation = context.GetArgument(target, Conv2DTranspose.Dilation);
        var padding = context.GetArgument(target, Conv2DTranspose.Padding);
        var outputPadding = context.GetArgument(target, Conv2DTranspose.OutputPadding);
        var groups = context.GetArgument(target, Conv2DTranspose.Groups);
        return IR.F.ShapeExpr.Conv2DTransposeShape(IR.F.Tensors.Cast(input, DataTypes.Int64), weights, stride, dilation, padding, outputPadding, groups);
    }
}
