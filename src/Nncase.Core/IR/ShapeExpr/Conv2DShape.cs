// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

public class Conv2DShape : ShapeExprOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Conv2DShape), 0, "input");

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(Conv2DShape), 1, "weights");

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public static readonly ParameterInfo Padding = new(typeof(Conv2DShape), 2, "padding", HasRank(2) & IsIntegral());

    /// <summary>
    /// Gets Stride.
    /// </summary>
    public static readonly ParameterInfo Stride = new(typeof(Conv2DShape), 3, "stride", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets Dilation.
    /// </summary>
    public static readonly ParameterInfo Dilation = new(typeof(Conv2DShape), 4, "dilation", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets Groups.
    /// </summary>
    public static readonly ParameterInfo Groups = new(typeof(Conv2DShape), 5, "groups", IsScalar() & IsIntegral());
}
