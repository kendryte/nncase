// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Nncase.Evaluator.EvaluatorUtil;

namespace Nncase.Evaluator.NN;
using static Tensorflow.Binding;

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
            _ => throw new ArgumentOutOfRangeException(nameof(pad.PadMode)),
        };
        return OrtKI.Pad(input, ToOnnxPadFormat(pads), constValue, mode).ToValue();
    }

    public IValue SymmetricPad(IEvaluateContext context, Pad pad)
    {
        var input = context.GetTFArgumentValue(pad, Pad.Input);
        var pads = context.GetTFArgumentValue(pad, Pad.Pads);
        var mode = "SYMMETRIC";
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
}
