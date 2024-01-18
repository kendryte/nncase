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
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnConv"/>.
/// </summary>
public class NcnnConvEvaluator : IEvaluator<NcnnConv>, ITypeInferencer<NcnnConv>, ICostEvaluator<NcnnConv>, IShapeEvaluator<NcnnConv>, IMetricEvaluator<NcnnConv>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnConv conv)
    {
        var inputs = context.GetOrtArgumentValue(conv, NcnnConv.Input);
        var weights = new Tensor<float>(conv.WeightData, new[] { conv.NumOutput, conv.WeightDataSize / (conv.KernelH * conv.KernelW), conv.KernelH, conv.KernelW }).ToOrtTensor();
        var bias = new Tensor<float>(conv.BiasData, new[] { conv.NumOutput }).ToOrtTensor();
        var result = OrtKI.Conv(inputs, weights, bias, "NOTSET", new long[] { conv.DilationH, conv.DilationW }, 1, new long[] { conv.KernelH, conv.KernelW }, new long[] { conv.PadLeft, conv.PadTop, conv.PadRight, conv.PadBottom }, new long[] { conv.StrideH, conv.StrideW });
        return OrtKI.Clip(result, conv.ActivationParams[0], conv.ActivationParams[1]).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnConv target)
    {
        var inputs = context.CheckArgumentType<TensorType>(target, NcnnConv.Input);
        return Visit(inputs, target);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnConv target)
    {
        var inputType = context.GetArgumentType<IRType>(target, NcnnConv.Input);
        var weightsType = new TensorType(DataTypes.Float32, new[] { target.WeightDataSize });
        var biasType = new TensorType(DataTypes.Float32, new[] { target.NumOutput });
        var macPerElement = (2 * target.WeightDataSize / target.NumOutput) - 1;

        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, (uint)macPerElement),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnConv target) => Metric.Zero;

    public Expr Visit(IShapeEvaluateContext context, NcnnConv target) => context.GetArgumentShape(target, NcnnConv.Input);

    private TensorType GetTensorType(IRType input) => input switch
    {
        TensorType t => t,
        DistributedType d => d.TensorType,
        _ => throw new InvalidCastException(),
    };

    private IRType Visit(TensorType input, NcnnConv conv)
    {
        var outputShape = input.Shape.ToList();
        var kernelShape = new[] { conv.NumOutput, conv.WeightDataSize / (conv.NumOutput * conv.KernelH * conv.KernelW), conv.KernelH, conv.KernelW };
        outputShape[0] = conv.NumOutput;

        outputShape[1] = IR.TypePatternUtility.GetWindowedOutputSize(
                input.Shape[1].FixedValue + conv.PadLeft + conv.PadRight,
                kernelShape[2],
                conv.StrideH,
                conv.DilationH,
                false);
        outputShape[2] = IR.TypePatternUtility.GetWindowedOutputSize(
                input.Shape[2].FixedValue + conv.PadTop + conv.PadBottom,
                kernelShape[3],
                conv.StrideW,
                conv.DilationW,
                false);
        return new TensorType(GetTensorType(input).DType, outputShape.ToArray());
    }
}
