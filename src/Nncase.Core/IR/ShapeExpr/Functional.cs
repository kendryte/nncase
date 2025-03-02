// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.ShapeExpr;

namespace Nncase.IR.F;

public static class ShapeExpr
{
    public static Call GetPaddings(Expr input, Expr weights, Expr strides, Expr dilation, Expr same, Expr lower) => new(new GetPaddings(), input, weights, strides, dilation, same, lower);
}
