// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Pad"/>.
/// </summary>
public class PadEvaluator : IEvaluator<Pad>, ITypeInferencer<Pad>, ICostEvaluator<Pad>, IShapeEvaluator<Pad>, IMetricEvaluator<Pad>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Pad pad)
    {
        var input = context.GetOrtArgumentValue(pad, Pad.Input);
        var pads = context.GetInt64OrtTensorArgumentValue(pad, Pad.Pads);
        var constValue = context.GetOrtArgumentValue(pad, Pad.Value);

        var mode = pad.PadMode switch
        {
            PadMode.Constant => "constant",
            PadMode.Reflect => "reflect",
            PadMode.Symmetric => "reflect",
            PadMode.Edge => "edge",
            _ => throw new ArgumentOutOfRangeException(nameof(pad)),
        };

        // when HasBindedMixQuantInfo is true, eval will do simulation of quant/dequant for some inputs, this is used for evaluate accumulated quant error for layers.
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            var pattern = IsRangeOfMarker(IsWildcard(), IsWildcard());
            if (pattern.MatchLeaf(context.CurrentCall.Arguments.ToArray()[0]) && ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo?.HasBindedMixQuantInfo == true)
            {
                var quantParam = ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo!.QuantParameter;

                // input feature map quantParam count should be 1 since input feature map quant is by tensor.
                Trace.Assert(quantParam.Count == 1);
                var inputFloat = input.ToArray<float>();
                for (var i = 0; i < inputFloat.Length; i++)
                {
                    var inputBufQuant = (double)((inputFloat[i] / (double)quantParam[0].Scale) + quantParam[0].ZeroPoint);
                    if (!(quantParam[0].Scale == 1.0f && quantParam[0].ZeroPoint == 0))
                    {
                        inputBufQuant = System.Math.Round((double)(float)inputBufQuant);
                    }

                    var inputBufDeQuant = (float)((inputBufQuant - quantParam[0].ZeroPoint) * (double)quantParam[0].Scale);
                    inputFloat[i] = (float)inputBufDeQuant;
                }

                input = OrtKISharp.Tensor.MakeTensor(inputFloat, input.Shape);
            }
        }

        if (pad.PadMode == PadMode.Symmetric)
        {
            return SymmetricPad(input, ToOnnxPadFormat(pads), constValue).ToValue();
        }
        else
        {
            return OrtKI.Pad(input, ToOnnxPadFormat(pads), constValue, mode).ToValue();
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Pad target)
    {
        var input = context.CheckArgumentType<IRType>(target, Pad.Input);
        var paddings = context.GetArgument(target, Pad.Pads);
        var padValue = context.GetArgument(target, Pad.Value);
        return input switch
        {
            DistributedType distributedType => Visit(distributedType, paddings, padValue),
            TensorType tensorType => TypeInference.PadType(tensorType, paddings, padValue),
            _ => new InvalidType("The pad input type not support"),
        };
    }

    public IRType Visit(DistributedType input, Expr paddings, Expr padValue)
    {
        if (TypeInference.PadType(input.TensorType, paddings, padValue) is not TensorType tensorType)
        {
            return new InvalidType("pad infer type failed");
        }

        return new DistributedType(tensorType, input.NdSBP, input.Placement);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Pad target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Pad.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Pad target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Pad.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Pad target)
    {
        var inShape = context.GetArgumentShape(target, Pad.Input);
        _ = context.GetArgumentRank(target, Pad.Input);
        var pads = context.GetArgument(target, Pad.Pads);
        var front = Slice(pads, new[] { 0 }, new[] { 1 }, new[] { 1 }, new[] { 1 });
        var end = Slice(pads, new[] { 1 }, new[] { 2 }, new[] { 1 }, new[] { 1 });

        // paddings = [4, 2] -> [4, 1] + [4, 1]
        var paddings = Cast(front + end, DataTypes.Int64);

        // outShape = inShape + paddings
        var padsSumShape = StackScalar(ShapeOf(paddings)[0]);
        var outShape = inShape + Reshape(paddings, padsSumShape);
        return outShape;
    }

    private OrtKISharp.Tensor SymmetricPad(OrtKISharp.Tensor input, long[] pads, OrtKISharp.Tensor constValue)
    {
        // Currently there isn't a symmetric padding mode in ONNX so we add a dummy row then use the reflect mode
        // and remove the dummy row with compress.Ex: 1234-> 012340-> 2101234043-> 21123443.Only do this to
        // dims with non - zero pads(if pads are constant)
        var rank = input.Rank;
        var nonZeroAxes = new List<int>();
        var incPads = new long[rank * 2];
        for (int i = 0; i < rank; i++)
        {
            if (pads[i] != 0 || pads[i + rank] != 0)
            {
                nonZeroAxes.Add(i);
                incPads[i] = 1;
                incPads[i + rank] = 1;
            }
        }

        var paddedInput = OrtKI.Pad(input, incPads, constValue, "constant");
        var output = OrtKI.Pad(paddedInput, pads, constValue, "reflect");
        foreach (var axis in nonZeroAxes)
        {
            var originLen = (int)input.Shape[axis];
            var leftPad = (int)pads[axis];
            var indices = Enumerable.Range(0, leftPad).Concat(Enumerable.Range(leftPad + 1, originLen)).Concat(Enumerable.Range(leftPad + originLen + 2, (int)pads[axis + rank])).ToArray();
            output = OrtKI.Gather(output, indices, axis);
        }

        return output;
    }
}
