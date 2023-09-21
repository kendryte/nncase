// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.ShapeExpr;

namespace Nncase.IR.F;

public static class ShapeExpr
{
    public static Call BroadcastShape(Expr[] inputs) => new(new BroadcastShape(), new IR.Tuple(inputs));

    public static Call Conv2DShape(Expr input, Expr weights, Expr padding, Expr stride, Expr dilation, Expr groups) => new(new Conv2DShape(), input, weights, padding, stride, dilation, groups);

    public static Call Conv2DTransposeShape(Expr input, Expr weights, Expr stride, Expr dilation, Expr padding, Expr outputPadding, Expr groups) => new(new Conv2DTransposeShape(), input, weights, stride, dilation, padding, outputPadding, groups);

    public static Call MatMulShape(Expr lhs, Expr rhs) => new(new MatMulShape(), lhs, rhs);

    public static Call GetPaddings(Expr input, Expr weights, Expr strides, Expr dilation, Expr same, Expr lower) => new(new GetPaddings(), input, weights, strides, dilation, same, lower);

    public static Call ReshapeShape(Expr input, Expr shape) => new(new ReshapeShape(), input, shape);

    public static Call SqueezeShape(Expr input, Expr dims) => new(new SqueezeShape(), input, dims);

    public static Call TransposeShape(Expr input, Expr perm) => new(new TransposeShape(), input, perm);

    public static Call UnsqueezeShape(Expr lhs, Expr rhs) => new(new UnsqueezeShape(), lhs, rhs);
}
