// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Imaging;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Evaluator for <see cref="ResizeImage"/>.
/// </summary>
public class ResizeImageEvaluator : IEvaluator<ResizeImage>, ITypeInferencer<ResizeImage>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ResizeImage target)
    {
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ResizeImage target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ResizeImage.Input);
        var newSize = context.GetArgument(target, ResizeImage.NewSize);
        return TypeInference.ResizeType(input, newSize, null);
    }
}
