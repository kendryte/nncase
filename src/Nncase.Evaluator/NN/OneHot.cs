// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="OneHot"/>.
/// </summary>
public class OneHotEvaluator : IEvaluator<OneHot>, ITypeInferencer<OneHot>, ICostEvaluator<OneHot>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, OneHot oneHot)
    {
        return OnnxOneHot(context, oneHot);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, OneHot target)
    {
        var indices = context.CheckArgumentType<TensorType>(target, OneHot.Indices);
        var values = context.CheckArgumentType<TensorType>(target, OneHot.Values);
        return Visit(context, target, indices, values);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, OneHot target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IValue OnnxOneHot(IEvaluateContext context, OneHot oneHot)
    {
        var indices = context.GetArgumentValueAsTensor<long>(oneHot, OneHot.Indices).ToOrtTensor();
        var depth = context.GetInt64OrtTensorArgumentValue(oneHot, OneHot.Depth);
        var values = context.GetArgumentValueAsTensor(oneHot, OneHot.Values);
        var axis = context.GetArgumentValueAsScalar<long>(oneHot, OneHot.Axis);

        var onnxValues = values.ElementType == DataTypes.Float32 ? values.ToOrtTensor()
            : values.Cast<float>().ToOrtTensor();

        // Set negative indices to depth + 1.
        if (oneHot.OneHotMode == OneHotMode.Normal)
        {
            indices = OrtKI.Where(OrtKI.Less(indices, 0L), depth, indices);
        }

        return new TensorValue(OrtKI.OneHot(indices, depth, onnxValues, axis).ToTensor().CastTo(values.ElementType));
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
