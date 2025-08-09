// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class VectorizedSoftmaxEvaluator : ITypeInferencer<VectorizedSoftmax>, ICostEvaluator<VectorizedSoftmax>, IEvaluator<VectorizedSoftmax>
{
    public IRType Visit(ITypeInferenceContext context, VectorizedSoftmax target)
    {
        var input = context.CheckArgumentType<IRType>(target, VectorizedSoftmax.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    public Cost Visit(ICostEvaluateContext context, VectorizedSoftmax target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IValue Visit(IEvaluateContext context, VectorizedSoftmax target)
    {
        var input = context.GetOrtArgumentValue(target, VectorizedSoftmax.Input);
        var shape = input.Shape.Select(i => (int)i).ToArray();
        OrtKISharp.Tensor softmax;
        if (!target.VectorizedAxes.Any(i => i == target.Axis))
        {
            softmax = OrtKI.Softmax(input, target.Axis);
        }
        else
        {
            var vectorizedAxis = shape.Length - target.VectorizedAxes.Count + target.VectorizedAxes.IndexOf(target.Axis);
            var max = OrtKI.ReduceMax(input, new long[] { target.Axis, vectorizedAxis }, 1);
            var exp = OrtKI.Exp(input - max);
            var reduceSum = OrtKI.ReduceSum(exp, new long[] { target.Axis, vectorizedAxis }, 1, 0);
            softmax = OrtKI.Div(exp, reduceSum);
        }

        return Value.FromTensor(Tensor.FromBytes(new TensorType(new VectorType(input.DataType.ToDataType(), shape.TakeLast(target.VectorizedAxes.Count).ToArray()), shape.SkipLast(target.VectorizedAxes.Count).ToArray()), softmax.BytesBuffer.ToArray()));
    }

    private IRType Visit(ITypeInferenceContext context, VectorizedSoftmax target, TensorType input)
    {
        foreach (var axis in target.VectorizedAxes)
        {
            if (axis >= input.Shape.Rank)
            {
                return new InvalidType("axis out of range");
            }
        }

        return input;
    }

    private IRType Visit(ITypeInferenceContext context, VectorizedSoftmax target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        return new DistributedType(tensorType, input.AxisPolicies, input.Placement);
    }
}
