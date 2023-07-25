// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Store"/>.
/// </summary>
public class StoreEvaluator : ITypeInferencer<Store>, IOpPrinter<Store>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Store target)
    {
        var handle = context.CheckArgumentType<TensorType>(target, Store.Handle);
        var index = context.CheckArgumentType<TensorType>(target, Store.Index);
        var value = context.CheckArgumentType<TensorType>(target, Store.Value);
        return Visit(target, handle, index, value);
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Store target, bool iLmode)
    {
        var handle = context.GetArgument(target, Store.Handle);
        _ = context.GetArgument(target, Store.Value);
        var index = context.GetArgument(target, Store.Index);
        return $"{handle}[{index}] = {index}";
    }

    private IRType Visit(Store target, TensorType handle, TensorType index, TensorType value)
    {
        if (handle.DType != value.DType)
        {
            return new InvalidType($"You Can't Load The {value.DType} To {handle.DType}");
        }

        return TupleType.Void;
    }
}
