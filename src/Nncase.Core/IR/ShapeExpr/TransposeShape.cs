// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

/// <summary>
/// Gets input.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class TransposeShape : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo InputShape = new(typeof(TransposeShape), 0, "input");

    /// <summary>
    /// Gets perm.
    /// </summary>
    public static readonly ParameterInfo Perm = new(typeof(TransposeShape), 1, "perm", HasRank(1) & IsIntegral());
}
