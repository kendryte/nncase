// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

public class MatMulShape : ShapeExprOp
{
    public static readonly ParameterInfo Lhs = new(typeof(MatMulShape), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(MatMulShape), 1, "rhs");
}
