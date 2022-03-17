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
namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="OneHot"/>.
/// </summary>
public class OneHotEvaluator : IEvaluator<OneHot>, ITypeInferencer<OneHot>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, OneHot oneHot)
    {
        return oneHot.OneHotMode == OneHotMode.Normal
            ? OnnxOneHot(context, oneHot)
            : TFOneHot(context, oneHot);
    }

    private IValue OnnxOneHot(IEvaluateContext context, OneHot oneHot)
    {
        var indices = context.GetArgumentValueAsTensor<long>(oneHot, OneHot.Indices);
        var depth = context.GetInt64OrtTensorArgumentValue(oneHot, OneHot.Depth);
        var values = context.GetOrtArgumentValue(oneHot, OneHot.Values);
        var axis = context.GetArgumentValueAsScalar<long>(oneHot, OneHot.Axis);
        return OrtKI.OneHot(indices.ToOrtTensor(), depth, values, axis).ToValue();
    }
    
    private IValue TFOneHot(IEvaluateContext context, OneHot oneHot)
    {
        var depth = context.GetArgumentValueAsScalar<int>(oneHot, OneHot.Depth);
        var indices = context.GetTFArgumentValue(oneHot, OneHot.Indices);
        var values = context.GetTFArgumentValue(oneHot, OneHot.Values);
        var axis = context.GetArgumentValueAsScalar<int>(oneHot, OneHot.Axis);
        return TF_OneHot(
            indices,
            ops.convert_to_tensor(depth),
            values[1],
            values[0],
            TF_DataType.TF_FLOAT,
            axis);
    }
    
    private static IValue TF_OneHot(
        Tensorflow.Tensor indices,
        Tensorflow.Tensor depth,
        Tensorflow.Tensor on_value,
        Tensorflow.Tensor off_value,
        TF_DataType dtype = TF_DataType.DtInvalid,
        int axis = -1,
        string name = "")
    {
        return tf_with(
            ops.name_scope(name, nameof(TF_OneHot), new
            {
                indices,
                depth,
                dtype,
            }),
            scope =>
            {
                TF_DataType tfDataType1 = TF_DataType.DtInvalid;
                TF_DataType tfDataType2 = TF_DataType.DtInvalid;
                if (dtype == TF_DataType.DtInvalid)
                {
                    dtype = TF_DataType.TF_FLOAT;
                }

                on_value = ops.convert_to_tensor(on_value, dtype, nameof(on_value));
                off_value = ops.convert_to_tensor(off_value, dtype, name = nameof(off_value));
                return gen_array_ops.one_hot(indices, depth, on_value, off_value, axis: axis, name: name);
            }).ToValue();
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
