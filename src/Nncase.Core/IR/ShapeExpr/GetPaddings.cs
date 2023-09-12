// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.ShapeExpr;

[PatternFunctionalGenerator]
public class GetPaddings : Op
{
    /// <summary>
    /// Gets Input.
    /// </summary>
    public static readonly ParameterInfo InputShape = new(typeof(GetPaddings), 0, "input_shape");

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo WeightsShape = new(typeof(GetPaddings), 1, "weights_shape");

    /// <summary>
    /// Gets Strides.
    /// </summary>
    public static readonly ParameterInfo Strides = new(typeof(GetPaddings), 2, "strides");

    /// <summary>
    /// Gets Dilations.
    /// </summary>
    public static readonly ParameterInfo Dilations = new(typeof(GetPaddings), 3, "dilations");

    /// <summary>
    /// Gets Same.
    /// </summary>
    public static readonly ParameterInfo Same = new(typeof(GetPaddings), 4, "same");

    /// <summary>
    /// Gets Lower.
    /// </summary>
    public static readonly ParameterInfo Lower = new(typeof(GetPaddings), 5, "lower");
}
