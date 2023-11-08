// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for BufferOf.
/// </summary>
[TypeInferGenerator]
public partial class BufferStoreEvaluator : ITypeInferencer<BufferStore>, IOpPrinter<BufferStore>
{
    public string Visit(IIRPrinterContext context, BufferStore target, bool iLmode)
    {
        if (iLmode)
        {
            throw new System.NotSupportedException();
        }

        return $"{context.GetArgument(target, BufferStore.Input)}[{context.GetArgument(target, BufferStore.Indices)}] = {context.GetArgument(target, BufferStore.Value)}";
    }

    private IRType Visit(TensorType input, TupleType indices, TensorType value)
    {
        if (indices.Count != input.Shape.Rank)
        {
            return new InvalidType($"the input buffer rank {input.Shape.Rank} != indices.Count {indices.Count}");
        }

        foreach (var item in indices)
        {
            if (item is not TensorType { IsScalar: true, DType: var dtype } || dtype != DataTypes.Int32)
            {
                return new InvalidType("indices is not int32 type!");
            }
        }

        if (!value.IsScalar || input.DType != value.DType)
        {
            return new InvalidType("value can't store!");
        }

        return TupleType.Void;
    }
}
