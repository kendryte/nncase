// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// SpaceToBatch expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class SpaceToBatch : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(SpaceToBatch), 0, "input");

    /// <summary>
    /// Gets block shape.
    /// </summary>
    public static readonly ParameterInfo BlockShape = new(typeof(SpaceToBatch), 1, "block_shape", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets paddings.
    /// </summary>
    public static readonly ParameterInfo Paddings = new(typeof(SpaceToBatch), 2, "paddings", HasShape(new[] { Dimension.Unknown, 2 }) & IsIntegral());
}
