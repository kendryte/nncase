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
/// Evaluator for <see cref="Conv2DTranspose"/>.
/// </summary>
public class Conv2DTransposeEvaluator : IEvaluator<Conv2DTranspose>, ITypeInferencer<Conv2DTranspose>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Conv2DTranspose conv)
    {
        var input = context.GetTorchArgumentValue(conv, Conv2DTranspose.Input);
        var weights = context.GetTorchArgumentValue(conv, Conv2DTranspose.Weights);
        var bias = context.GetTorchArgumentValue(conv, Conv2DTranspose.Bias);
        var stride = context.GetArgumentValue(conv, Conv2DTranspose.Stride).ToArray<long>();

        // [w:[left right] h:[top bottom]]
        var pad = context.GetArgumentValue(conv, Conv2DTranspose.Padding).ToTensor<long>();
        var dilation = context.GetArgumentValue(conv, Conv2DTranspose.Dilation).ToArray<long>();
        var groups = context.GetArgumentValue(conv, Conv2DTranspose.Groups).ToScalar<long>();
        if (conv.PadMode != PadMode.Constant)
        {
            throw new NotImplementedException($"Conv2DTranspose with {conv.PadMode}!");
        }

        // var afterPad = torchF.pad(input, new long[] { pad[0, 0], pad[1, 0], pad[0, 1], pad[1, 1] });
        return torchF.conv_transpose2d(
            input,
            weights,
            bias,
            stride,
            dilation: dilation,
            groups: groups).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2DTranspose target)
    {
        if (context.GetArgument(target, Conv2DTranspose.OutputShape) is Const outShapeValue)
        {
            var input = context.CheckArgumentType<TensorType>(target, Conv2D.Input);
            return new TensorType(input.DType, new Shape(outShapeValue.ToTensor<int>()));
        }
        else
        {
            return new InvalidType("Conv2dTranspose can't infer shape with dynamic outputShape");
        }
    }
}
