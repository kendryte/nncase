// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Range = Nncase.IR.Tensors.Range;

namespace Nncase.IR.F;

/// <summary>
/// NN functional helper.
/// </summary>
public static class Tensors
{
    public static Call Transpose(Expr input, Expr perm) => new Call(new Transpose(), input, perm);

    public static Expr NHWCToNCHW(Expr input) => Transpose(input, new[] { 0, 3, 1, 2 });

    public static Expr NCHWToNHWC(Expr input) => Transpose(input, new[] { 0, 2, 3, 1 });

    public static Call Broadcast(Expr input, Expr shape) => new Call(new Broadcast(), input, shape);

    public static Call Cast(Expr input, DataType newType) => new Call(new Cast(newType), input);

    public static Call Concat(Tuple input, Expr axis) => new Call(new Concat(), input, axis);

    public static Call CumSum(Expr input, Expr axis, Expr exclusive, Expr reverse) => new Call(new CumSum(), input, axis, exclusive, reverse);

    public static Call Expand(Expr input, Expr shape) => new Call(new Expand(), input, shape);

    public static Call Flatten(Expr input, Expr axis) => new Call(new Flatten(), input, axis);

    public static Call Gather(Expr input, Expr axis, Expr index) => new Call(new Gather(), input, axis, index);

    public static Call GatherND(Expr input, Expr batch_dims, Expr index) => new Call(new GatherND(), input, batch_dims, index);

    public static Call MatMul(Expr input, Expr other) => new Call(new MatMul(), input, other);

    public static Call Prod(Expr input) => new Call(new Prod(), input);

    public static Call Range(Expr begin, Expr end, Expr step) => new Call(new Range(), begin, end, step);

    public static Call Reduce(ReduceOp reduceOp, Expr input, Expr axis, Expr initValue, Expr keepDims) => new Call(new Reduce(reduceOp), input, axis, initValue, keepDims);

    public static Call ReduceArg(ReduceArgOp reduceArgOp, Expr input, Expr axis, Expr keepDims, Expr selectLastIndex) => new Call(new ReduceArg(reduceArgOp), input, axis, keepDims, selectLastIndex);

    public static Call ReduceMean(Expr input, Expr axis, Expr initValue, Expr keepDims) => Reduce(ReduceOp.Mean, input, axis, initValue, keepDims);

    public static Call ReduceMin(Expr input, Expr axis, Expr initValue, Expr keepDims) => Reduce(ReduceOp.Min, input, axis, initValue, keepDims);

    public static Call ReduceMax(Expr input, Expr axis, Expr initValue, Expr keepDims) => Reduce(ReduceOp.Min, input, axis, initValue, keepDims);

    public static Call ReduceSum(Expr input, Expr axis, Expr initValue, Expr keepDims) => Reduce(ReduceOp.Sum, input, axis, initValue, keepDims);

    public static Call Reshape(Expr input, Expr shape) => new Call(new Reshape(), input, shape);

    public static Call ReverseSequence(Expr input, Expr seqLens, Expr batchAxis, Expr timeAxis) =>
        new Call(new ReverseSequence(), input, seqLens, batchAxis, timeAxis);

    public static Call ShapeOf(Expr input) => new Call(new ShapeOf(), input);

    /// https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice
    public static Call Slice(Expr input, Expr begins, Expr ends, Expr axes, Expr strides) =>
      new Call(new Slice(), input, begins, ends, axes, strides);

    public static Call Slice(Expr input, Expr begins, Expr ends, int rank)
    {
        var axes = Const.FromSpan<int>(Enumerable.Range(0, rank).ToArray());
        var strides = axes with { Data = new IRBytes(DataTypes.GetBytes<int>(Enumerable.Repeat(1, rank).ToArray())) };
        return new Call(new Slice(), input, begins, ends, axes, strides);
    }

    public static Expr SliceIndex(Expr input, int index) => Slice(input, new[] { index }, new[] { index + 1 }, 1);

    public static Expr SizeOf(Expr input) => new Call(new SizeOf(), input);

    public static Call Stack(Expr inputs, Expr axis) => new Call(new Stack(), inputs, axis);

    /// squeeze input by give dims
    public static Call Squeeze(Expr input, Expr dims) => new Call(new Squeeze(), input, dims);

    public static Call Unsqueeze(Expr input, Expr dims) => new Call(new Unsqueeze(), input, dims);

    public static Call Quantize(Expr input, Expr zeroPoint, Expr scale, DataType targetType) => new Call(new Quantize(targetType), input, zeroPoint, scale);

    public static Call DeQuantize(Expr input, Expr zeroPoint, Expr scale, DataType targetType) => new Call(new Dequantize(targetType), input, zeroPoint, scale);

    // return a scalar
    public static Expr Rank(Expr input) => Slice(ShapeOf(ShapeOf(input)), new[] { 0 }, new[] { 1 }, 1);

    // sections (int or list[int])
    public static Call Split(Expr input, Expr axis, Expr sections) => new Call(new Split(), input, axis, sections);
}
