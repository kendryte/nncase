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
        var value = context.GetArgument(target, Store.Value);
        var index = context.GetArgument(target, Store.Index);
        return $"{handle}[{index}] = {value}";
    }

    private IRType Visit(Store target, TensorType handle, TensorType index, TensorType value)
    {
        if (handle.DType is not PointerType { ElemType: DataType elemType } || elemType != value.DType)
        {
            return new InvalidType($"You Can't Load The {value.DType} To {handle.DType}");
        }

        if (index.DType != DataTypes.Int32)
        {
            return new InvalidType($"store value type {index.DType} not supported");
        }

        return TupleType.Void;
    }
}
