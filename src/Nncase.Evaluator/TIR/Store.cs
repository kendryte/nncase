// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Store"/>.
/// </summary>
public class StoreEvaluator : ITypeInferencer<Store>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Store target)
    {
        var handle = context.CheckArgumentType<HandleType>(target, Store.Handle);
        var index = context.CheckArgumentType<TensorType>(target, Store.Index);
        var value = context.CheckArgumentType<TensorType>(target, Store.Value);
        return Visit(target, handle, index, value);
    }

    private IRType Visit(Store target, HandleType handle, TensorType index, TensorType value)
    {
        var lanes = index.IsScalar ? 1 : index.Shape[0].FixedValue;
        if (handle.DType != value.DType)
        {
            return new InvalidType($"You Can't Load The {value.DType} To {handle.DType}");
        }

        if (value.DType.Lanes != lanes)
        {
            return new InvalidType($"You're Index Lanes {lanes} Is Not Equal Value Lanes {handle.DType.Lanes}");
        }

        return TupleType.Void;
    }
}
