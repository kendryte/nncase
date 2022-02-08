// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Conv2D"/>.
/// </summary>
public class Conv2DEvaluator : IEvaluator<Conv2D>, ITypeInferencer<Conv2D>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Conv2D conv)
    {
        var input = context.GetTorchArgumentValue(conv, Conv2D.Input);
        var weights = context.GetTorchArgumentValue(conv, Conv2D.Weights);
        var bias = context.GetTorchArgumentValue(conv, Conv2D.Bias);
        var stride = context.GetArgumentValue(conv, Conv2D.Stride).ToTensor<long>();

        // [w:[left right] h:[top bottom]]
        var pad = context.GetArgumentValue(conv, Conv2D.Padding).ToTensor<long>();
        var dilation = context.GetArgumentValue(conv, Conv2D.Dilation).ToTensor<long>();
        var groups = context.GetArgumentValue(conv, Conv2D.Groups).ToScalar<long>();
        if (conv.PadMode != PadMode.Constant)
        {
            throw new NotImplementedException($"Conv2D with {conv.PadMode}!");
        }

        // pad in TorchSharp will reorder
        // when pad.Count == 4, [0, 2, 1, 3]
        // order should be passed: left top right bottom
        var afterPad = torchF.pad(input, new long[] { pad[0, 0], pad[1, 0], pad[0, 1], pad[1, 1] });
        return torchF.conv2d(
            afterPad,
            weights,
            bias,
            strides: new long[] { stride[0], stride[1] },
            dilation: new long[] { dilation[0], dilation[1] },
            groups: groups).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2D target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Conv2D.Input);
        var weights = context.CheckArgumentType<TensorType>(target, Conv2D.Weights);
        return Visit(context, target, input, weights);
    }

    private IRType Visit(ITypeInferenceContext context, Conv2D target, TensorType input, TensorType weights)
    {
        var args = context.GetArguments(target, Conv2D.Stride, Conv2D.Padding, Conv2D.Dilation, Conv2D.Groups);
        return TypeInference.Conv2DType(input, weights, args[0], args[1], args[2], args[3]);
    }
}
