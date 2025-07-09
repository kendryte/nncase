﻿// Copyright (c) Canaan Inc. All rights reserved.
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
    public static Call Transpose(Expr input, Shape perm) => new Call(new Transpose(), input, perm);

    public static Expr NHWCToNCHW(Expr input)
    {
        int[] perm;
        if (input.CheckedShape.Rank == 4)
        {
            perm = new[] { 0, 3, 1, 2 };
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

    public static Call Broadcast(Expr input, Shape shape) => new Call(new Broadcast(), input, shape);

    public static Call Bitcast(PrimType type, Expr input, PrimType newType, Expr shape) =>
        new Call(new Bitcast(type, newType), input, shape);

    public static Call Cast(Expr input, DataType newType, CastMode castMode = CastMode.KDefault, IRArray<int> packAxes = default) =>
        new Call(new Cast(newType, castMode, packAxes), input);

    public static Call Concat(BaseExpr input, int axis) => new Call(new Concat(axis), input);

    public static Call ConstantOfShape(Shape shape, Expr value) => new Call(new ConstantOfShape(), shape, value);

    public static Call CumSum(Expr input, Dimension axis, Expr exclusive, Expr reverse) =>
        new Call(new CumSum(), input, axis, exclusive, reverse);

    public static Call Expand(Expr input, Shape shape) => new Call(new Expand(), input, shape);

    public static Call Flatten(Expr input, Dimension axis) => new Call(new Flatten(), input, axis);

    public static Call Gather(Expr input, int axis, Expr index) => new Call(new Gather(axis), input, index);

    public static Call GatherElements(Expr input, Expr axis, Expr indices) =>
        new Call(new GatherElements(), input, axis, indices);

    public static Call GatherND(Expr input, Expr batch_dims, Expr index) =>
        new Call(new GatherND(), input, batch_dims, index);

    public static Call ScatterND(Expr input, Expr indices, Expr updates) =>
        new Call(new ScatterND(), input, indices, updates);

    public static Call MatMul(Expr input, Expr other) => new Call(new MatMul(DataTypes.Float32), input, other);

    public static Call MatMul(Expr input, Expr other, DataType outDataType) => new Call(new MatMul(outDataType), input, other);

    public static Call Prod(Expr input)
    {
        return Reduce(ReduceOp.Prod, input, Enumerable.Range(0, input.CheckedShape.Rank).Select(x => (long)x).ToArray(), IR.F.Tensors.Cast(1, input.CheckedDataType, CastMode.KDefault), false);
    }

    public static Call Range(Expr begin, Expr end, Expr step) => new Call(new Range(), begin, end, step);

    public static Call Reduce(ReduceOp reduceOp, Expr input, Shape axes, Expr initValue, Expr keepDims) =>
        new Call(new Reduce(reduceOp), input, axes, initValue, keepDims);

    public static Call ReduceArg(ReduceArgOp reduceArgOp, DataType destType, Expr input, Dimension axis, Expr keepDims, Expr selectLastIndex) =>
        new Call(new ReduceArg(reduceArgOp, destType), input, axis, keepDims, selectLastIndex);

    public static Call ReduceMean(Expr input, Shape axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Mean, input, axis, initValue, keepDims);

    public static Call ReduceMin(Expr input, Shape axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Min, input, axis, initValue, keepDims);

    public static Call ReduceMax(Expr input, Shape axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Max, input, axis, initValue, keepDims);

    public static Call ReduceSum(Expr input, Shape axis, Expr initValue, Expr keepDims) =>
        Reduce(ReduceOp.Sum, input, axis, initValue, keepDims);

    public static Call Reshape(Expr input, Shape shape) => new Call(new Reshape(), input, shape);

    public static Call ReverseSequence(Expr input, Expr seqLens, Expr batchAxis, Expr timeAxis) =>
        new Call(new ReverseSequence(), input, seqLens, batchAxis, timeAxis);

    public static Call ShapeOf(Expr input) => new Call(new ShapeOf(), input);

    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice
    public static Call Slice(Expr input, Shape begins, Shape ends, Shape axes, Shape strides) =>
        new Call(new Slice(), input, begins, ends, axes, strides);

    public static Call Slice(Expr input, Shape begins, Shape ends, int rank)
    {
        if (rank < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(rank));
        }

        var axes = Tensor.FromRange(0L, rank).ToArray();
        var strides = Tensor.FromScalar(1L, rank).ToArray();
        return Slice(input, begins, ends, axes, strides);
    }

    public static Expr SliceIndex(Expr input, int index) => Slice(input, new[] { index }, new[] { index + 1 }, 1);

    public static Expr SizeOf(Expr input) => new Call(new SizeOf(), input);

    public static Call Stack(BaseExpr inputs, Dimension axis) => new Call(new Stack(), inputs, axis);

    // squeeze input by give dims
    public static Call Squeeze(Expr input, Shape dims) => new Call(new Squeeze(), input, dims);

    public static Call Unsqueeze(Expr input, Shape dims) => new Call(new Unsqueeze(), input, dims);

    // return a scalar
    public static Call Rank(Expr input) => GetItem(ShapeOf(ShapeOf(input)), 0);

    // sections (int or list[int])
    public static Call Split(Expr input, Dimension axis, Shape sections) => new Call(new Split(), input, axis, sections);

    public static Call Tile(Expr input, Shape repeats) => new Call(new Tile(), input, repeats);

    public static Call Where(Expr cond, Expr x, Expr y, bool isTfWhere = false) => new Call(new Where(isTfWhere), cond, x, y);

    /// <summary>
    /// get item from the input.
    /// </summary>
    /// <param name="input">input.</param>
    /// <param name="index">index.</param>
    /// <returns>call.</returns>
    public static Call GetItem(BaseExpr input, Dimension index) => new Call(new GetItem(), input, index);

    public static Call GetItem(BaseExpr input, Shape index) => new Call(new GetItem(), input, index);

    public static Call TopK(Expr x, Expr k, Expr axis, Expr largest, Expr sorted) =>
        new Call(new TopK(), x, k, axis, largest, sorted);

    public static Call IndexOf(Expr input, Expr value) => new Call(new IndexOf(), input, value);

    public static Call Trilu(Expr input, Expr k, Expr upper) => new Call(new Trilu(), input, k, upper);

    public static Expr Pack(Expr input, int[] lanes, int[] axes)
    {
        if (lanes.Length != axes.Length)
        {
            throw new NotSupportedException();
        }

        if (axes.Length == 0)
        {
            return input;
        }

        return new Call(new Pack(lanes, axes), input);
    }

    public static Expr PackMask(Expr input, MaskVectorStyle style, int elementBits, int lanes, int axis)
    {
        return new Call(new PackMask(style, elementBits, lanes, axis), input);
    }

    public static Expr Unpack(Expr input, int[] lanes, int[] axes)
    {
        if (lanes.Length != axes.Length)
        {
            throw new NotSupportedException();
        }

        if (axes.Length == 0)
        {
            return input;
        }

        return new Call(new Unpack(lanes, axes), input);
    }
}
