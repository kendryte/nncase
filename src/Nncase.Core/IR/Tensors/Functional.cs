// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.Diagnostics;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Range = Nncase.IR.Tensors.Range;

namespace Nncase.IR.F;

/// <summary>
/// NN functional helper.
/// </summary>
public static class Tensors
{
    public static Call Transpose(Expr input, Expr perm) => new Call(new Transpose(), input, perm);

    public static Expr NHWCToNCHW(Expr input)
    {
        int[] perm;
        if (input.CheckedShape.Rank == 4)
        {
            perm = new[] { 0, 3, 1, 2 };
        }
        else if (input.CheckedShape.Rank == 3)
        {
            perm = new[] { 2, 0, 1 };
        }
        else
        {
            throw new InvalidOperationException();
        }

        return Transpose(input, perm);
    }

    public static Expr NCHWToNHWC(Expr input)
    {
        int[] perm;
        if (input.CheckedShape.Rank == 4)
        {
            perm = new[] { 0, 2, 3, 1 };
        }
        else if (input.CheckedShape.Rank == 3)
        {
            perm = new[] { 0, 2, 1 };
        }
        else
        {
            throw new InvalidOperationException();
        }

        return Transpose(input, perm);
    }

    public static Expr NHWCToWNCH(Expr input) => Transpose(input, new[] { 2, 0, 3, 1 });

    public static Call Broadcast(Expr input, Expr shape) => new Call(new Broadcast(), input, shape);

    public static Call Bitcast(PrimType type, Expr input, PrimType newType, Expr shape) =>
        new Call(new Bitcast(type, newType), input, shape);

    public static Call Cast(Expr input, DataType newType, CastMode castMode = CastMode.KDefault) =>
        new Call(new Cast(newType, castMode), input);

    public static Call Concat(Expr input, Expr axis) => new Call(new Concat(), input, axis);

    public static Call ConstantOfShape(Expr shape, Expr value) => new Call(new ConstantOfShape(), shape, value);

    public static Call CumSum(Expr input, Expr axis, Expr exclusive, Expr reverse) =>
        new Call(new CumSum(), input, axis, exclusive, reverse);

    public static Call Expand(Expr input, Expr shape)
    {
        if (shape.InferenceType() && shape.CheckedShape.IsScalar)
        {
            shape = Unsqueeze(shape, new[] { 0 });
        }

        return new Call(new Expand(), input, shape);
    }

    public static Call Flatten(Expr input, Expr axis) => new Call(new Flatten(), input, axis);

    public static Call Gather(Expr input, Expr axis, Expr index) => new Call(new Gather(), input, axis, index);

    public static Call GatherElements(Expr input, Expr axis, Expr indices) =>
        new Call(new GatherElements(), input, axis, indices);

    public static Call GatherND(Expr input, Expr batch_dims, Expr index) =>
        new Call(new GatherND(), input, batch_dims, index);

    public static Call ScatterND(Expr input, Expr indices, Expr updates) =>
        new Call(new ScatterND(), input, indices, updates);

    public static Call MatMul(Expr input, Expr other) => new Call(new MatMul(), input, other);

    public static Call Prod(Expr input)
    {
        return Reduce(ReduceOp.Prod, input, Enumerable.Range(0, input.CheckedShape.Rank).Select(x => (long)x).ToArray(), IR.F.Tensors.Cast(1, input.CheckedDataType, CastMode.KDefault), false);
    }

    public static Call Range(Expr begin, Expr end, Expr step) => new Call(new Range(), begin, end, step);

    public static Call Reduce(ReduceOp reduceOp, Expr input, Expr axis, Expr initValue, Expr keepDims) =>
        new Call(new Reduce(reduceOp), input, axis, initValue, keepDims);

    public static Call ReduceArg(ReduceArgOp reduceArgOp, PrimType destType, Expr input, Expr axis, Expr keepDims, Expr selectLastIndex) =>
        new Call(new ReduceArg(reduceArgOp, destType), input, axis, keepDims, selectLastIndex);

    public static Call ReduceMean(Expr input, Expr axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Mean, input, axis, initValue, keepDims);

    public static Call ReduceMin(Expr input, Expr axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Min, input, axis, initValue, keepDims);

    public static Call ReduceMax(Expr input, Expr axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Max, input, axis, initValue, keepDims);

    public static Call ReduceSum(Expr input, Expr axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Sum, input, axis, initValue, keepDims);

    public static Call Reshape(Expr input, Expr shape) => new Call(new Reshape(), input, shape);

    public static Call ReverseSequence(Expr input, Expr seqLens, Expr batchAxis, Expr timeAxis) =>
        new Call(new ReverseSequence(), input, seqLens, batchAxis, timeAxis);

    public static Call ShapeOf(Expr input) => new Call(new ShapeOf(), input);

    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice
    public static Call Slice(Expr input, Expr begins, Expr ends, Expr axes, Expr strides) =>
        new Call(new Slice(), input, begins, ends, axes, strides);

    public static Call Slice(Expr input, Expr begins, Expr ends, int rank)
    {
        if (rank < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(rank));
        }

        var axes = Tensor.FromRange(0, rank);
        var strides = Tensor.FromScalar(1, rank);
        return new Call(new Slice(), input, begins, ends, axes, strides);
    }

    public static Expr SliceIndex(Expr input, int index) => Slice(input, new[] { index }, new[] { index + 1 }, 1);

    public static Expr SizeOf(Expr input) => new Call(new SizeOf(), input);

    public static Call Stack(Expr inputs, Expr axis) => new Call(new Stack(), inputs, axis);

    // squeeze input by give dims
    public static Call Squeeze(Expr input, Expr dims) => new Call(new Squeeze(), input, dims);

    public static Call Unsqueeze(Expr input, Expr dims) => new Call(new Unsqueeze(), input, dims);

    // return a scalar
    public static Expr Rank(Expr input) => Slice(ShapeOf(ShapeOf(input)), new[] { 0 }, new[] { 1 }, 1);

    // sections (int or list[int])
    public static Call Split(Expr input, Expr axis, Expr sections) => new Call(new Split(), input, axis, sections);

    public static Call Tile(Expr input, Expr repeats) => new Call(new Tile(), input, repeats);

    public static Call Where(Expr cond, Expr x, Expr y, bool isTfWhere = false) => new Call(new Where(isTfWhere), cond, x, y);

    /// <summary>
    /// get item from the input.
    /// </summary>
    /// <param name="input">input.</param>
    /// <param name="index">index.</param>
    /// <returns>call.</returns>
    public static Call GetItem(Expr input, Expr index) => new Call(new GetItem(), input, index);

    public static Call StackScalar(Expr scalar) => Stack(new Tuple(scalar), 0);

    public static Call TopK(Expr x, Expr k, Expr axis, Expr largest, Expr sorted) =>
        new Call(new TopK(), x, k, axis, largest, sorted);

    public static Call IndexOf(Expr input, Expr value) => new Call(new IndexOf(), input, value);

    public static Call Trilu(Expr input, Expr k, Expr upper) => new Call(new Trilu(), input, k, upper);
}
