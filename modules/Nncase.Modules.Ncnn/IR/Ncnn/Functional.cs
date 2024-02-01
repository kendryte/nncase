// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
    public static Call NcnnConv(Expr expr, float[] weightsData, float[] biasData, int numOutput, int[] kernel, int[] dilation, int[] strides, int[] pads, int biasTerm, int weightsDataSize, int int8Flag, int actType, float[] actParams, float padValue, int dynamicFlag) =>
        new Call(new NcnnConv(weightsData, biasData, numOutput, kernel[1], kernel[0], dilation[1], dilation[0], strides[1], strides[0], pads[0], pads[1], pads[2], pads[3], padValue, biasTerm, weightsDataSize, int8Flag, actType, actParams, dynamicFlag), expr);

    public static Call NcnnCumsum(Expr expr, int axis) => new Call(new NcnnCumsum(axis), expr);

    public static Call NcnnElu(Expr expr, float alpha) => new Call(new NcnnElu(alpha), expr);

    public static Call NcnnErf(Expr expr) => new Call(new NcnnErf(), expr);

    public static Call NcnnHardSigmoid(Expr expr, float alpha, float beta) => new Call(new NcnnHardSigmoid(alpha, beta), expr);

    public static Call NcnnHardSwish(Expr expr, float alpha, float beta) => new Call(new NcnnHardSwish(alpha, beta), expr);

    public static Call NcnnInstanceNorm(Expr expr, int channels, float eps, int affine, float[] gammaData, float[] betaData) =>
        new Call(new NcnnInstanceNorm(channels, eps, affine, gammaData, betaData), expr);

    public static Call NcnnLRN(Expr expr, float alpha, float beta, float bias, int size) => new Call(new NcnnLRN(alpha, beta, bias, size), expr);

    public static Call NcnnLSTM(Expr expr, int outputSize, int hiddenSize, int weightDataSize, int direction, float[] w, float[] b, float[] r) =>
        new Call(new NcnnLSTM(outputSize, hiddenSize, weightDataSize, direction, w, b, r), expr);
}
