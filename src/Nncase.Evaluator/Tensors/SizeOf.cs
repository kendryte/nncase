// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="SizeOf"/>.
/// </summary>
public class SizeOfEvaluator : IEvaluator<SizeOf>, ITypeInferencer<SizeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, SizeOf size)
    {
        var input = context.GetTorchArgumentValue(size, SizeOf.Input);
        var v = (Tensor)(int)input.numel();
        return Value.FromTensor(v);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, SizeOf target)
    {
        return new TensorType(DataTypes.Int32, Shape.Scalar);
    }
}
