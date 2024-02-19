// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class PackedSoftMaxEvaluator : ITypeInferencer<PackedSoftMax>, ICostEvaluator<PackedSoftMax>, IEvaluator<PackedSoftMax>
{
    public IRType Visit(ITypeInferenceContext context, PackedSoftMax target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedSoftMax.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    public Cost Visit(ICostEvaluateContext context, PackedSoftMax target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IValue Visit(IEvaluateContext context, PackedSoftMax target)
    {
        var input = context.GetOrtArgumentValue(target, PackedSoftMax.Input);
        var shape = input.Shape.Select(i => (int)i).ToArray();
        if (!target.PackedAxes.Any(i => i == target.Axis))
        {
            var softmax = OrtKI.Softmax(input, target.Axis);
            return Value.FromTensor(Tensor.FromBytes(new TensorType(new VectorType(input.DataType.ToDataType(), shape.TakeLast(target.PackedAxes.Count).ToArray()), shape.SkipLast(target.PackedAxes.Count).ToArray()), softmax.BytesBuffer.ToArray()));
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    private IRType Visit(ITypeInferenceContext context, PackedSoftMax target, TensorType input)
    {
        foreach (var axis in target.PackedAxes)
        {
            if (axis >= input.Shape.Rank)
            {
                return new InvalidType("axis out of range");
            }
        }

        return input;
    }

    private IRType Visit(ITypeInferenceContext context, PackedSoftMax target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        return new DistributedType(tensorType, input.NdSBP, input.Placement);
    }
}
