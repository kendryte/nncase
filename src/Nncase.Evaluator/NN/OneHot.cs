// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
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
    public Const Visit(EvaluatorContext context, OneHot oneHot)
    {
        var depth = context.GetArgumentConstScalar<int>(oneHot, OneHot.Depth);
        var rawIndices = context.GetTFArgument(oneHot, OneHot.Indices);
        var afterIndices = rawIndices.ToConst().ToArray<int>().Select(x => x < 0 ? x + depth : x).ToArray();
        var indices = new NDArray(afterIndices, rawIndices.shape);
        var onValue = context.GetTFArgument(oneHot, OneHot.OnValue);
        var offValue = context.GetTFArgument(oneHot, OneHot.OffValue);
        var axis = context.GetArgumentConstScalar<int>(oneHot, OneHot.Axis);
        return TF_OneHot(
            indices,
            ops.convert_to_tensor(depth),
            onValue,
            offValue,
            TF_DataType.TF_FLOAT,
            axis).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, OneHot target)
    {
        var indices = context.CheckArgumentType<TensorType>(target, OneHot.Indices);
        var onValue = context.CheckArgumentType<TensorType>(target, OneHot.OnValue);
        return Visit(context, target, indices, onValue);
    }

    private static Tensor TF_OneHot(
        Tensor indices,
        Tensor depth,
        Tensor on_value,
        Tensor off_value,
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
                tfDataType1 = dtype;
                off_value = ops.convert_to_tensor(off_value, dtype, name = nameof(off_value));
                tfDataType2 = dtype;
                return gen_array_ops.one_hot(indices, depth, on_value, off_value, axis: axis, name: name);
            });
    }

    private IRType Visit(ITypeInferenceContext context, OneHot target, TensorType indices, TensorType onValue)
    {
        // indices_shape[:axis] + [depth] + indices_shape[axis:]
        if (context.GetArgument(target, OneHot.Axis) is Const axisValue
            && context.GetArgument(target, OneHot.Depth) is Const depthValue)
        {
            var newShape = indices.Shape.InsertAndClone(axisValue.ToScalar<int>(), depthValue.ToScalar<int>());
            return new TensorType(onValue.DType, newShape);
        }

        return new InvalidType("OneHot axis or depth is not const");
    }
}
