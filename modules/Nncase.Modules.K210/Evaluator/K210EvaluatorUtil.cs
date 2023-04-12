// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using Shape = Nncase.IR.Shape;

namespace Nncase.Evaluator;

public static class K210EvaluatorUtil
{
    public static T BoxCast<T>(float v, DataType dt)
    {
        if (dt == DataTypes.Float32)
        {
            return (T)(object)v;
        }
        else if (dt == DataTypes.BFloat16)
        {
            return (T)(object)BFloat16.RoundToBFloat16(v);
        }
        else
        {
            throw new InvalidOperationException("510 Evaluator Cast only support float to T(float / bfloat16)");
        }
    }

    public static IValue Act(Tensor input, Tensor act, Tensor fusedClamp)
    {
        var inShape = input.Shape.ToValueArray();
        var batchSize = ComputeSize(input.Shape.ToValueArray()[1..]);
        var innerSize = ComputeSize(input.Shape.ToValueArray()[2..]);
        var data = input.ToArray<float>();
        var output = new BFloat16[ComputeSize(input.Shape)];
        var actData = act.ToArray<float>();
        var clamp = fusedClamp.ToArray<float>();
        for (int b = 0; b < inShape[0]; b++)
        {
            for (int i = 0; i < inShape[1]; i++)
            {
                for (int j = 0; j < innerSize; j++)
                {
                    var index = (b * batchSize) + (i * innerSize) + j;
                    var actBegin = i * 5;
                    var v = ApplyActivation(data[index], actData[actBegin..(actBegin + 5)], clamp);
                    output[index] = BFloat16.RoundToBFloat16(v);
                }
            }
        }

        return Value.FromTensor(Tensor.From<BFloat16>(output, input.Shape));
    }

    public static Const FakeAct(Tensor input, Tensor act, Tensor fusedClamp)
    {
        var outerSize = ComputeSize(input.Shape.ToValueArray()[..1]);
        var innerSize = ComputeSize(input.Shape.ToValueArray()[2..]);
        var data = input.ToArray<float>();
        var output = new float[ComputeSize(input.Shape)];
        var actData = act.ToArray<float>();
        var clamp = fusedClamp.ToArray<float>();
        for (int i = 0; i < outerSize; i++)
        {
            for (int j = 0; j < innerSize; j++)
            {
                var index = (i * innerSize) + j;
                output[index] = ApplyActivation(data[index], actData[i..(i + 5)], clamp);
            }
        }

        return Const.FromTensor(Tensor.From<float>(output, input.Shape));
    }

    public static float ApplyActivation(float value, float[] clamp)
    {
        return System.Math.Clamp(value, clamp[0], clamp[1]);
    }

    public static float ApplyActivation(float value, float[] act, float[] clamp)
    {
        return ApplyActivation(ApplyGNNEActivation(value, act), clamp);
    }

    public static float ApplyGNNEActivation(float value, float x0, float kl, float bl, float kr, float br)
    {
        return value < x0
            ? (value * kl) + bl
            : (value * kr) + br;
    }

    public static float ApplyGNNEActivation(float value, float[] act)
    {
        return ApplyGNNEActivation(value, act[0], act[1], act[2], act[3], act[4]);
    }

    public static int ComputeSize(Const input)
    {
        return ComputeSize(input.CheckedShape);
    }

    public static int ComputeSize(int[] shape)
    {
        return shape.Aggregate(1, (sum, x) => sum * x);
    }

    public static int ComputeSize(Shape shape)
    {
        return shape.Prod().FixedValue;
    }

    public static int Linear_index(int[] shape, int[] index)
    {
        int newIndex = index[0];
        for (int i = 1; i < shape.Length; i++)
        {
            newIndex = (newIndex * shape[i]) + index[i];
        }

        return newIndex;
    }

    public static int[] GetDefaultStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        var dataSize = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = dataSize;
            dataSize = strides[i] * shape[i];
            if (shape[i] == 1)
            {
                strides[i] = 0;
            }
        }

        return strides;
    }

    public static Const ConcatOutput<T>(List<T[]> outputTmp, int[] outShape)
      where T : unmanaged, IEquatable<T>
    {
        var init = Array.Empty<T>().AsEnumerable();
        var outputData = outputTmp.Aggregate(init, (sum, output) => sum.Concat(output)).ToArray();
        return Const.FromTensor(Tensor.From<T>(outputData, outShape));
    }

    public static OrtKISharp.Tensor ProcFakePSum(IValue psum, int oc)
    {
        return psum == Value.None
            ? Enumerable
                .Repeat(0, oc)
                .Select(x => (float)x)
                .ToArray()
            : psum.AsTensor().ToOrtTensor();
    }

    public static OrtKISharp.Tensor DefaultBias(IValue psum, int oc)
    {
        return Tensor.FromArray(Enumerable
            .Repeat(0, oc)
            .Select(x => BFloat16.FromRaw(0))
            .ToArray<BFloat16>()).ToOrtTensor();
    }

    /// <summary>
    /// nncase pads format to onnx pads format.
    /// </summary>
    public static long[] ToOnnxPadFormat(OrtKISharp.Tensor pads)
    {
        if (pads.Rank != 2)
        {
            throw new InvalidOperationException($"Pad's rank must be 2, but get {pads.Rank}");
        }

        return OrtKI.Transpose(pads, new long[] { 1, 0 }).ToArray<long>();
    }
}
