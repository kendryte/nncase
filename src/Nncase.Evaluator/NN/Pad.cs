// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Nncase.Evaluator.EvaluatorUtil;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Tensorflow.Binding;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Pad"/>.
/// </summary>
public class PadEvaluator : IEvaluator<Pad>, ITypeInferencer<Pad>, ICostEvaluator<Pad>, IShapeEvaluator<Pad>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Pad pad)
    {
        var input = context.GetOrtArgumentValue(pad, Pad.Input);
        var pads = context.GetInt64OrtTensorArgumentValue(pad, Pad.Pads);
        var constValue = context.GetOrtArgumentValue(pad, Pad.Value);
        if (pad.PadMode == PadMode.Symmetric)
        {
            var result = SymmetricPad(context, pad);
            return result;
        }

        var mode = pad.PadMode switch
        {
            PadMode.Constant => "constant",
            PadMode.Reflect => "reflect",
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

        return OrtKI.Pad(input, ToOnnxPadFormat(pads), constValue, mode).ToValue();
    }

    public IValue SymmetricPad(IEvaluateContext context, Pad pad)
    {
        var input = context.GetTFArgumentValue(pad, Pad.Input);
        var pads = context.GetTFArgumentValue(pad, Pad.Pads);
        var mode = "SYMMETRIC";

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

                input = tf.constant(inputFloat, TF_DataType.TF_FLOAT, input.shape);
            }
        }

        var result = tf.Context.ExecuteOp(
            "MirrorPad",
            null!,
            new ExecuteOpArgs(input, pads).SetAttributes(new { mode }))[0];
        return result.ToValue();
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

    public Expr Visit(IShapeEvaluateContext context, Pad target)
    {
        var inShape = context.GetArgumentShape(target, Pad.Input);
        var rank = context.GetArgumentRank(target, Pad.Input);
        var pads = context.GetArgument(target, Pad.Pads);
        var front = Slice(pads, new[] {0}, new[] { 1 }, new[] {1}, new[] {1});
        var end = Slice(pads, new[] {1}, new[] { 2 }, new[] {1}, new[] {1});
        // paddings = [4, 2] -> [4, 1] + [4, 1]
        var paddings = front + end;
        // outShape = inShape + paddings
        var outShape = inShape + Reshape(paddings, rank);
        DumpScope.Current.DumpIR(outShape, "paddings");
        return outShape;
    }
}
