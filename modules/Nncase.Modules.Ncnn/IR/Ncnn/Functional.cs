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
}
