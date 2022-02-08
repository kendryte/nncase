// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Transpose"/>.
/// </summary>
public class TransposeEvaluator : IEvaluator<Transpose>, ITypeInferencer<Transpose>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, Transpose tr)
    {
        var input = context.GetTorchArgument(tr, Transpose.Input);
        var perm = context.GetArgumentConst(tr, Transpose.Perm);
        return input.permute(perm.ToTensor<long>().ToArray()).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Split.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Transpose target, TensorType input)
    {
        var permExpr = context.GetArgument(target, Transpose.Perm);
        return TypeInference.TransposeType(input, permExpr);
    }
}
