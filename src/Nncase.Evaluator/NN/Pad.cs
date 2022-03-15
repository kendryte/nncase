// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Pad"/>.
/// </summary>
public class PadEvaluator : IEvaluator<Pad>, ITypeInferencer<Pad>
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
            PadMode.Symmetric => throw new NotImplementedException(),
            PadMode.Edge => "edge",
            _ => throw new ArgumentOutOfRangeException(nameof(pad.PadMode)),
        };
        return OrtKI.Pad(input, pads, constValue, mode).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Pad target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Pad.Input);
        var paddings = context.GetArgument(target, Pad.Pads);
        var padValue = context.GetArgument(target, Pad.Value);
        return TypeInference.PadType(input, paddings, padValue);
    }
}
