// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="OneHot"/>.
/// </summary>
public class OneHotEvaluator : IEvaluator<OneHot>, ITypeInferencer<OneHot>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, OneHot oneHot)
    {
        var indices = context.GetOrtArgumentValue(oneHot, OneHot.Indices);
        var depth = context.GetOrtArgumentValue(oneHot, OneHot.Depth);
        var values = context.GetOrtArgumentValue(oneHot, OneHot.Values);
        var axis = context.GetArgumentValueAsScalar<long>(oneHot, OneHot.Axis);
        return OrtKI.OneHot(indices, depth, values, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, OneHot target)
    {
        var indices = context.CheckArgumentType<TensorType>(target, OneHot.Indices);
        var values = context.CheckArgumentType<TensorType>(target, OneHot.Values);
        return Visit(context, target, indices, values);
    }
    
    private IRType Visit(ITypeInferenceContext context, OneHot target, TensorType indices, TensorType values)
    {
        // indices_shape[:axis] + [depth] + indices_shape[axis:]
        if (context.GetArgument(target, OneHot.Axis) is TensorConst axisValue
            && context.GetArgument(target, OneHot.Depth) is TensorConst depthValue)
        {
            var newShape = indices.Shape.InsertAndClone(axisValue.Value.ToScalar<int>(), depthValue.Value.ToScalar<int>());
            return new TensorType(values.DType, newShape);
        }

        return new InvalidType("OneHot axis or depth is not const");
    }
}
