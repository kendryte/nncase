// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

/// <summary>
/// Reshape expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ReshapeShape : Op
{
    /// <summary>
    /// Gets input shape.
    /// </summary>
    public static readonly ParameterInfo InputShape = new(typeof(ReshapeShape), 0, "input_shape");

    /// <summary>
    /// Gets shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(ReshapeShape), 1, "shape", HasRank(1));
}
