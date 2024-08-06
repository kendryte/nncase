// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.ArgsStruct;
using Nncase.IR.Ncnn;

namespace Nncase.IR.F;

public sealed class Ncnn
{
    public static Call NcnnUnary(Expr expr, UnaryOperationType unaryOp) =>
        new Call(new NcnnUnary(unaryOp), expr);

    public static Call NcnnSoftmax(Expr expr, int axis) =>
        new Call(new NcnnSoftmax(axis), expr);

    public static Call NcnnBatchNorm(Expr expr, int channels, float eps, float[] slopeData, float[] meanData, float[] varData, float[] biasData) =>
        new Call(new NcnnBatchNorm(channels, eps, slopeData, meanData, varData, biasData), expr);

    public static Call NcnnBinary(Expr[] inExpr, BinaryOperationType binaryOp, int lorR, float[]? constInput, int[]? constShape) =>
        new Call(new NcnnBinary(binaryOp, lorR, constInput, constShape), inExpr);

    public static Call NcnnCelu(Expr expr, float alpha) =>
        new Call(new NcnnCelu(alpha), expr);

    public static Call NcnnClip(Expr expr, float min, float max) =>
        new Call(new NcnnClip(min, max), expr);

    public static Call NcnnConcat(Expr[] expr, int axis) =>
        new Call(new NcnnConcat(axis), new IR.Tuple(expr));

    // In ncnn param file, lauout is [w, h] for kernel, dilation
    public static Call NcnnConv(Expr expr, ConvArgs args) =>
        new Call(new NcnnConv(args), expr);

    public static Call NcnnCumsum(Expr expr, int axis) => new Call(new NcnnCumsum(axis), expr);

    public static Call NcnnElu(Expr expr, float alpha) => new Call(new NcnnElu(alpha), expr);

    public static Call NcnnErf(Expr expr) => new Call(new NcnnErf(), expr);

    public static Call NcnnHardSigmoid(Expr expr, float alpha, float beta) => new Call(new NcnnHardSigmoid(alpha, beta), expr);

    public static Call NcnnHardSwish(Expr expr, float alpha, float beta) => new Call(new NcnnHardSwish(alpha, beta), expr);

    public static Call NcnnInstanceNorm(Expr expr, int channels, float eps, int affine, float[] gammaData, float[] betaData) =>
        new Call(new NcnnInstanceNorm(channels, eps, affine, gammaData, betaData), expr);

    public static Call NcnnLayerNorm(Expr expr, int affineSize, float eps, int affine, float[] gammaData, float[] betaData) =>
        new Call(new NcnnLayerNorm(affineSize, eps, affine, gammaData, betaData), expr);

    public static Call NcnnLRN(Expr expr, float alpha, float beta, float bias, int size) => new Call(new NcnnLRN(alpha, beta, bias, size), expr);

    public static Call NcnnLSTM(Expr expr, int outputSize, int hiddenSize, int weightDataSize, int direction, float[] w, float[] b, float[] r) =>
        new Call(new NcnnLSTM(outputSize, hiddenSize, weightDataSize, direction, w, b, r), expr);

    public static Call NcnnPadding(Expr expr, int top, int bottom, int left, int right, int type, float value, int front, int behind) =>
        new Call(new NcnnPadding(top, bottom, left, right, type, value, front, behind), expr);

    public static Call NcnnPooling(Expr expr, PoolingArgs poolingArgs) =>
        new Call(new NcnnPooling(poolingArgs), expr);

    public static Call NcnnPReLU(Expr expr, float[] slope) =>
        new Call(new NcnnPReLU(slope), expr);

    public static Call NcnnReduction(Expr expr, ReductionArgs reductionArgs) =>
        new Call(new NcnnReduction(reductionArgs), expr);

    public static Call NcnnReshape(Expr expr, int[] shape) =>
        new Call(new NcnnReshape(shape), expr);

    public static Call NcnnSELU(Expr expr, float alpha, float gamma) =>
        new Call(new NcnnSELU(alpha, gamma), expr);

    public static Call NcnnSigmoid(Expr expr) =>
        new Call(new NcnnSigmoid(), expr);

    public static Call NcnnCrop(Expr expr, CropArgs args) =>
        new Call(new NcnnCrop(args), expr);

    public static Call NcnnSoftplus(Expr expr) =>
        new Call(new NcnnSoftplus(), expr);

    public static Call NcnnSlice(Expr expr, int[] slices, int axis) =>
        new Call(new NcnnSlice(slices, axis), expr);

    public static Call NcnnTile(Expr expr, int[] repeats) =>
        new Call(new NcnnTile(repeats), expr);

    public static Call NcnnPermute(Expr expr, int orderType, int[] perm) =>
        new Call(new NcnnPermute(orderType, perm), expr);

    public static Call NcnnMatMul(Expr[] inExpr, int lorR, float[]? constInput, int[]? constShape) =>
        new Call(new NcnnMatMul(lorR, constInput, constShape), inExpr);

    public static Call NcnnConvTranspose(Expr expr, ConvTransposeArgs args) => new Call(new NcnnConvTranspose(args), expr);

    public static Call NcnnCast(Expr expr, int fromType, int toType) =>
        new Call(new NcnnCast(fromType, toType), expr);

    public static Call NcnnGELU(Expr expr) => new Call(new NcnnGELU(), expr);

    public static Call NcnnDequantize(Expr expr, float[] scale, float[] bias) => new Call(new NcnnDequantize(scale, bias), expr);

    public static Call NcnnSqueeze(Expr expr, int[] dims) => new Call(new NcnnSqueeze(dims), expr);

    public static Call NcnnUnsqueeze(Expr expr, int[] dims) => new Call(new NcnnUnsqueeze(dims), expr);
}
