// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

[PatternFunctionalGenerator]
public sealed partial class UnsqueezeShape : Op
{
    /// <summary>
    /// Gets input_shape.
    /// </summary>
    public static readonly ParameterInfo InputShape = new(typeof(UnsqueezeShape), 0, "input_shape");

    /// <summary>
    /// Gets dimension.
    /// </summary>
    public static readonly ParameterInfo Dim = new(typeof(UnsqueezeShape), 1, "dim", HasRank(1) & IsIntegral());
}
