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
/// Evaluator for <see cref="NcnnConvTranspose"/>.
/// </summary>
public class NcnnConvTransposeEvaluator : IEvaluator<NcnnConvTranspose>, ITypeInferencer<NcnnConvTranspose>, ICostEvaluator<NcnnConvTranspose>, IShapeEvaluator<NcnnConvTranspose>, IMetricEvaluator<NcnnConvTranspose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnConvTranspose convTranspose)
    {
        var inputs = context.GetOrtArgumentValue(convTranspose, NcnnConvTranspose.Input);
        var weights = Transpose(convTranspose.Args.WeightData, new int[] { 1, 0, 2, 3 }).Evaluate().AsTensor().ToOrtTensor();
        var bias = new Tensor<float>(convTranspose.Args.BiasData, new[] { convTranspose.Args.NumOutput }).ToOrtTensor();
        var result = OrtKI.ConvTranspose(inputs, weights, bias, "NOTSET", new long[] { convTranspose.Args.DilationH, convTranspose.Args.DilationW }, 1, new long[] { convTranspose.Args.KernelH, convTranspose.Args.KernelW }, new long[] { convTranspose.Args.OutputPadBottom, convTranspose.Args.OutputPadRight, }, new long[] { convTranspose.Args.OutputH, convTranspose.Args.OutputW }, new long[] { convTranspose.Args.PadRight, convTranspose.Args.PadBottom, convTranspose.Args.PadRight, convTranspose.Args.PadBottom }, new long[] { convTranspose.Args.StrideH, convTranspose.Args.StrideW });
        return OrtKI.Clip(result, convTranspose.Args.ActivationParams[0], convTranspose.Args.ActivationParams[1]).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnConvTranspose target)
    {
        var inputs = context.CheckArgumentType<TensorType>(target, NcnnConvTranspose.Input);
        return Visit(inputs, target);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnConvTranspose target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnConvTranspose.Input);
        _ = inputType.Shape.ToValueArray();
        var weightsType = new TensorType(target.Args.WeightData.ElementType, target.Args.WeightData.Shape);
        var weightsShape = weightsType.Shape.ToValueArray();
        var biasType = new TensorType(target.Args.WeightData.ElementType, new[] { target.Args.BiasData.Length });

        var macPerElement = weightsShape[1] * weightsShape[2] * weightsShape[3];
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, (uint)macPerElement * 2),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnConvTranspose target)
    {
        var returnType = context.GetReturnType<TensorType>();
        _ = returnType.Shape.ToValueArray();

        var inputType = context.GetArgumentType<TensorType>(target, NcnnConvTranspose.Input);
        var inputShape = inputType.Shape.ToValueArray();
        var weightType = new TensorType(target.Args.WeightData.ElementType, target.Args.WeightData.Shape);
        var weightShape = weightType.Shape.ToValueArray();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = (UInt128)(inputShape[0] * weightShape[0] * weightShape[1] * inputShape[2] * inputShape[3] * weightShape[2] * weightShape[3]),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnConvTranspose target) => context.GetArgumentShape(target, NcnnConvTranspose.Input);

    private TensorType GetTensorType(IRType input) => input switch
    {
        TensorType t => t,
        DistributedType d => d.TensorType,
        _ => throw new InvalidCastException(),
    };

    private IRType Visit(TensorType input, NcnnConvTranspose convTranspose)
    {
        var outputShape = input.Shape.ToList();
        outputShape[0] = convTranspose.Args.NumOutput;
        outputShape[1] = convTranspose.Args.OutputH;
        outputShape[2] = convTranspose.Args.OutputW;

        return new TensorType(GetTensorType(input).DType, outputShape.ToArray());
    }
}
