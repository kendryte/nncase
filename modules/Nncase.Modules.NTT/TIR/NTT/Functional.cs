// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.TIR.F;

public partial class NTT
{
    /// <summary>
    /// the ptr of can create the *PtrName in the c code.
    /// </summary>
    /// <param name="name">c pointer name.</param>
    /// <param name="primType">type.</param>
    /// <returns>call.</returns>
    public static Call PtrOf(string name, DataType primType) => new Call(new PtrOf(name, primType));

    public static Call SramPtr(Expr input, DataType primType) => new Call(new SramPtr(primType), input);

    public static Call TensorLoad(Expr dest, Expr src, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TensorLoad(ndsbp, placement), dest, src);
    }

    public static Call TensorStore(Expr src, Expr dest, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TensorStore(ndsbp, placement), src, dest);
    }

    public static Call Unary(UnaryOp unaryOp, Expr input, Expr output)
    {
        return new Call(new TIR.NTT.Unary(unaryOp), input, output);
    }

    public static Call Binary(BinaryOp binaryOp, Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new TIR.NTT.Binary(binaryOp), lhs, rhs, output);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output, Expr loadC, IRArray<int> lhsPackedAxes, IRArray<int> rhsPackedAxes, bool transA = false, bool transB = false, bool fusedReduce = false, string cSourcePath = "")
    {
        return new Call(new Matmul(lhsPackedAxes, rhsPackedAxes, transA, transB, fusedReduce, cSourcePath), lhs, rhs, output, loadC);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output, Expr loadC)
    {
        return new Call(new Matmul(new IRArray<int>(), new IRArray<int>(), false, false, false, null), lhs, rhs, output, loadC);
    }

    public static Call SUMMA(Expr lhs, Expr rhs, Expr output, Expr loadC, IRArray<int> lhsPackedAxes, IRArray<int> rhsPackedAxes, bool transA = false, bool transB = false)
    {
        return new Call(new SUMMA(lhsPackedAxes, rhsPackedAxes, transA, transB), lhs, rhs, output, loadC);
    }

    public static Call SUMMA(Expr lhs, Expr rhs, Expr output, Expr loadC)
    {
        return new Call(new SUMMA(new IRArray<int>(), new IRArray<int>(), false, false), lhs, rhs, output, loadC);
    }

    public static Expr Pack(Expr input, Expr output, IRArray<int> lanes, IRArray<int> axes)
    {
        return new Call(new Pack(lanes, axes), input, output);
    }

    public static Call Conv2D(Expr input, Expr weights, Expr bias, Expr output, long[] stride, long[] padding, long[] dilation, long groups, PadMode padMode, DistributedType distributedType) => new Call(new Conv2D(stride, padding, dilation, groups, padMode, distributedType), input, weights, bias, output);

    public static Expr Unpack(Expr input, Expr output, IRArray<int> lanes, IRArray<int> axes)
    {
        return new Call(new Unpack(lanes, axes), input, output);
    }

    public static Expr PackedSoftmax(Expr input, Expr output, int axis, IRArray<int> packedAxes)
    {
        return new Call(new PackedSoftmax(axis, packedAxes), input, output);
    }

    public static Expr PackedLayerNorm(Expr input, Expr scale, Expr bias, Expr output, int axis, float epsilon, bool usemean, IRArray<int> packedAxes, IRArray<Dimension> padedNums)
    {
        return new Call(new PackedLayerNorm(axis, epsilon, usemean, packedAxes, padedNums, null!), input, scale, bias, output);
    }

    public static Expr InstanceNorm(Expr input, Expr scale, Expr bias, Expr output, float epsilon, IRArray<int> packedAxes, IRArray<Dimension> padedNums, DistributedType distributedType)
    {
        return new Call(new InstanceNorm(epsilon, packedAxes, padedNums, distributedType), input, scale, bias, output);
    }

    public static Expr PackedBinary(Expr lhs, Expr rhs, Expr output, BinaryOp binaryOp, IRArray<int> lhsPackedAxes, IRArray<Dimension> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<Dimension> rhsPadedNums)
    {
        return new Call(new PackedBinary(binaryOp, lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs, output);
    }

    public static Call ResizeImage(Expr input, Expr output, int[] packedAxes, Dimension[] padedNums, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return new Call(new ResizeImage(packedAxes, padedNums, newSize, resizeMode, transformationMode, nearestMode), input, output);
    }

    public static Expr Slice(Expr input, RankedShape begins, RankedShape ends, Expr ret, int[] axes, int[] strides)
    {
        return new Call(new Slice(axes, strides), input, begins, ends, ret);
    }

    public static Expr Concat(Expr[] inputs, Expr ret, int axis)
    {
        return new Call(new Concat(axis), inputs.Concat(new[] { ret }).ToArray());
    }

    public static Expr Reshape(Expr input, Expr ret)
    {
        return new Call(new Reshape(), input, ret);
    }

    public static Expr PagedAttention(Expr q, Expr kvcache, Expr extra, Expr scale, int layerId, Expr ret, IRArray<IR.NN.AttentionDimKind> layout)
    {
        return new Call(new PagedAttention(layerId, layout), q, kvcache, extra, scale, ret);
    }

    public static Expr UpdatePagedAttentionKVCache(Expr value, Expr kvcache, IR.NN.AttentionCacheKind kind, int layerId, IRArray<IR.NN.AttentionDimKind> layout)
    {
        return new Call(new UpdatePagedAttentionKVCache(kind, layerId, layout), value, kvcache);
    }

    public static Expr GatherPagedAttentionKVCache(Expr value, Expr kvcache, Expr output)
    {
        return new Call(new GatherPagedAttentionKVCache(), value, kvcache, output);
    }

    public static Expr CreatePagedAttentionKVCache(IR.NN.PagedAttentionConfig config, Expr numSeqs, Expr numTokens, Expr contextLens, Expr seqLens, Expr blockTable, Expr slotMapping, Expr numBlocks, Expr kvCaches, Expr output)
    {
        return new Call(new CreatePagedAttentionKVCache(config), numSeqs, numTokens, contextLens, seqLens, blockTable, slotMapping, numBlocks, kvCaches, output);
    }

    public static Expr IdentityPagedAttentionKVCache(Expr input, Expr numSeqs, Expr numTokens, Expr contextLens, Expr seqLens, Expr blockTable, Expr slotMapping, Expr numBlocks, Expr kvCaches)
    {
        return new Call(new IdentityPagedAttentionKVCache(), input, numSeqs, numTokens, contextLens, seqLens, blockTable, slotMapping, numBlocks, kvCaches);
    }

    public static Expr Swish(Expr buffer, Expr ret, float v)
    {
        return new Call(new Swish(v), buffer, ret);
    }

    public static Expr Gather(Expr input, Expr indcies, Expr ret, int axis)
    {
        return new Call(new Gather(axis), input, indcies, ret);
    }

    public static Expr GetItem(Expr input, BaseExpr index, Expr ret)
    {
        return new Call(new GetItem(), input, index, ret);
    }

    public static Expr Transpose(Expr buffer, Expr ret, int[] perm)
    {
        return new Call(new Transpose(perm), buffer, ret);
    }

    public static Expr Pad(Expr input, Expr ret, Dimension[] pads, float padValue)
    {
        return new Call(new Pad(pads, padValue), input, ret);
    }

    public static Expr Im2col(Expr input, Expr output, IRArray<long> kernel, IRArray<int> stride, IRArray<int> padding, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new Im2col(kernel, stride, padding, packedAxes, padedNums), input, output);
    }

    public static Expr Reduce(Expr input, Expr ret, Expr loadPrevious, int[] packedAxes, Dimension[] padedNums, IRArray<int> axis, bool keepDims, ReduceOp reduceOp)
    {
        return new Call(new TIR.NTT.Reduce(packedAxes, padedNums, axis, keepDims, reduceOp), input, ret, loadPrevious);
    }

    public static Expr ReduceArg(Expr input, Expr ret, int axis, bool keepDims, bool selectLastIndex, ReduceArgOp reduceArgOp, DataType destType)
    {
        return new Call(new TIR.NTT.ReduceArg(axis, keepDims, selectLastIndex, reduceArgOp, destType), input, ret);
    }

    public static Call GatherReduceScatter(Expr input, Expr output, DistributedType inType, DistributedType outType)
    {
        return new Call(new TIR.NTT.GatherReduceScatter(inType, outType), input, output);
    }

    public static Call Clamp(Expr input, Expr output, float min, float max)
    {
        return new Call(new TIR.NTT.Clamp(min, max), input, output);
    }

    public static Call Cast(Expr input, Expr output, DataType newType, CastMode castMode, IRArray<int> packAxes = default)
    {
        return new Call(new TIR.NTT.Cast(newType, castMode, packAxes.IsDefaultOrEmpty ? Array.Empty<int>() : packAxes), input, output);
    }

    public static Call Where(Expr cond, Expr x, Expr y, Expr output)
    {
        return new Call(new TIR.NTT.Where(), cond, x, y, output);
    }

    public static Call Expand(Expr input, Expr output)
    {
        return new Call(new TIR.NTT.Expand(), input, output);
    }

    public static Call Erf(Expr input, Expr output)
    {
        return new Call(new TIR.NTT.Erf(), input, output);
    }

    public static Call Compare(CompareOp compareOp, Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new TIR.NTT.Compare(compareOp), lhs, rhs, output);
    }

    public static Call ScatterND(Expr input, Expr indices, Expr updates, Expr output)
    {
        return new Call(new TIR.NTT.ScatterND(), input, indices, updates, output);
    }

    public static Expr Stack(Expr[] inputs, Expr ret, int axis)
    {
        return new Call(new Stack(axis), inputs.Concat(new[] { ret }).ToArray());
    }

    public static Expr ShapeOf(Expr inputs, Expr ret)
    {
        return new Call(new TIR.NTT.ShapeOf(), inputs, ret);
    }

    public static Expr ConstantOfShape(Shape shape, Expr value, Expr ret)
    {
        return new Call(new TIR.NTT.ConstantOfShape(), shape, value, ret);
    }

    public static Expr Range(Expr begin, Expr end, Expr step, Expr ret)
    {
        return new Call(new TIR.NTT.Range(), begin, end, step, ret);
    }

    public static Expr GetPositionIds(Expr kvCache, Expr ret)
    {
        return new Call(new TIR.NTT.GetPositionIds(), kvCache, ret);
    }
}
