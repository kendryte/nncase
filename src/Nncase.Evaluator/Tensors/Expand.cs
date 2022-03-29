// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;
using Shape = Nncase.IR.Shape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Expand"/>.
/// </summary>
public class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Expand expand)
    {
        var input = context.GetOrtArgumentValue(expand, Expand.Input);
        var shape = context.GetInt64OrtTensorArgumentValue(expand, Expand.Shape);
        return OrtKI.Expand(input, shape).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Expand target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Expand.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType input)
    {
        var shape = context.GetArgument(target, Expand.Shape);
        return shape switch
        {
            TensorConst constShape => new TensorType(input.DType, new Shape(constShape.Value.Cast<int>())),
            Call call => call.CheckedType is TensorType
                ? new TensorType(call.CheckedDataType, Shape.Unranked) 
                : new InvalidType(((InvalidType)call.CheckedType).Reason),
            Var var => new TensorType(var.CheckedDataType, Shape.Unranked),
            _ => throw new InvalidDataException("invalid shape")
        };
    }
}
