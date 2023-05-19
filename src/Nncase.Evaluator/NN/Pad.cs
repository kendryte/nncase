// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Pad"/>.
/// </summary>
public class PadEvaluator : IEvaluator<Pad>, ITypeInferencer<Pad>, ICostEvaluator<Pad>
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
        var input = context.CheckArgumentType<TensorType>(target, Pad.Input);
        var paddings = context.GetArgument(target, Pad.Pads);
        var padValue = context.GetArgument(target, Pad.Value);
        return TypeInference.PadType(input, paddings, padValue);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Pad target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Pad.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = outputType is TensorType outT ? CostUtility.GetMemoryAccess(outT) : CostUtility.GetMemoryAccess(inputType),
        };
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
