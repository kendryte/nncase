// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Ramp"/>.
/// </summary>
public class RampEvaluator : ITypeInferencer<Ramp>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Ramp target)
    {
        var offset = context.CheckArgumentType<TensorType>(target, Ramp.Offset);
        var stride = context.CheckArgumentType<TensorType>(target, Ramp.Stride);
        return Visit(target, offset, stride);
    }

    private IRType Visit(Ramp target, TensorType offset, TensorType stride)
    {
        // TODO maybe need simpify when the Lanes==1.
        return new TensorType(DataTypes.Int32, new Shape(target.Lanes));
    }
}
