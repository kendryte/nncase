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
        var weights = new Tensor<float>(conv.Args.WeightData, new[] { conv.Args.NumOutput, conv.Args.WeightDataSize / (conv.Args.KernelH * conv.Args.KernelW), conv.Args.KernelH, conv.Args.KernelW }).ToOrtTensor();
        var bias = new Tensor<float>(conv.Args.BiasData, new[] { conv.Args.NumOutput }).ToOrtTensor();
        var result = OrtKI.Conv(inputs, weights, bias, "NOTSET", new long[] { conv.Args.DilationH, conv.Args.DilationW }, 1, new long[] { conv.Args.KernelH, conv.Args.KernelW }, new long[] { conv.Args.PadLeft, conv.Args.PadTop, conv.Args.PadRight, conv.Args.PadBottom }, new long[] { conv.Args.StrideH, conv.Args.StrideW });
        return OrtKI.Clip(result, conv.Args.ActivationParams[0], conv.Args.ActivationParams[1]).ToValue();
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
        var weightsType = new TensorType(DataTypes.Float32, new[] { target.Args.WeightDataSize });
        var biasType = new TensorType(DataTypes.Float32, new[] { target.Args.NumOutput });
        var macPerElement = (2 * target.Args.WeightDataSize / target.Args.NumOutput) - 1;

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
        var kernelShape = new[] { conv.Args.NumOutput, conv.Args.WeightDataSize / (conv.Args.NumOutput * conv.Args.KernelH * conv.Args.KernelW), conv.Args.KernelH, conv.Args.KernelW };
        outputShape[0] = conv.Args.NumOutput;

        outputShape[1] = IR.TypePatternUtility.GetWindowedOutputSize(
                input.Shape[1].FixedValue + conv.Args.PadLeft + conv.Args.PadRight,
                kernelShape[2],
                conv.Args.StrideH,
                conv.Args.DilationH,
                false);
        outputShape[2] = IR.TypePatternUtility.GetWindowedOutputSize(
                input.Shape[2].FixedValue + conv.Args.PadTop + conv.Args.PadBottom,
                kernelShape[3],
                conv.Args.StrideW,
                conv.Args.DilationW,
                false);
        return new TensorType(GetTensorType(input).DType, outputShape.ToArray());
    }
}
