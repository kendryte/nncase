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

namespace Nncase.IR.Tensors;

/// <summary>
/// Slice expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Slice : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Slice), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets begins.
    /// </summary>
    public static readonly ParameterInfo Begins = new(typeof(Slice), 1, "begins", IsIntegral() & HasRank(1));

    /// <summary>
    /// Gets ends.
    /// </summary>
    public static readonly ParameterInfo Ends = new(typeof(Slice), 2, "ends", IsIntegral() & HasRank(1));

    /// <summary>
    /// Gets axes.
    /// </summary>
    public static readonly ParameterInfo Axes = new(typeof(Slice), 3, "axes", IsIntegral() & HasRank(1));

    /// <summary>
    /// Gets strides.
    /// </summary>
    public static readonly ParameterInfo Strides = new(typeof(Slice), 4, "strides", IsIntegral() & HasRank(1));
}
