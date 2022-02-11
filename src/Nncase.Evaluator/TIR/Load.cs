// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Load"/>.
/// </summary>
public class LoadEvaluator : ITypeInferencer<Load>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Load target)
    {
        var handle = context.CheckArgumentType<HandleType>(target, Load.Handle);
        var index = context.CheckArgumentType<TensorType>(target, Load.Index);
        return Visit(target, handle, index);
    }

    private IRType Visit(Load target, HandleType handle, TensorType index)
    {
        int lanes = index.IsScalar ? 1 : index.Shape[0].FixedValue;
        return new TensorType(handle.DType with { Lanes = lanes }, Shape.Scalar);
    }
}
