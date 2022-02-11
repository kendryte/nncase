// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Pad"/>.
/// </summary>
public class PadEvaluator : IEvaluator<Pad>, ITypeInferencer<Pad>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Pad pad)
    {
        var input = context.GetTFArgumentValue(pad, Pad.Input);
        var pads = context.GetTFArgumentValue(pad, Pad.Pads);
        var constant_values = context.GetArgumentValue(pad, Pad.Value).ToScalar<int>();
        var mode = pad.PadMode switch
        {
            PadMode.Constant => "CONSTANT",
            PadMode.Reflect => "REFLECT",
            PadMode.Symmetric => "SYMMETRIC",
            PadMode.Edge => "EDGE",
            _ => throw new ArgumentOutOfRangeException(nameof(pad.PadMode)),
        };
        return tf.Context.ExecuteOp(
            "Pad",
            null!,
            new ExecuteOpArgs(input, pads, mode, constant_values))[0].ToConst();

        // return tf.pad(input, pads, mode: mode, constant_values:constant_values).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Pad target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Pad.Input);
        var paddings = context.GetArgument(target, Pad.Pads);
        return TypeInference.PadType(input, paddings);
    }
}
