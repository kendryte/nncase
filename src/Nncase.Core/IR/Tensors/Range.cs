// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Range expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Range : Op
{
    /// <summary>
    /// Gets begin.
    /// </summary>
    public static readonly ParameterInfo Begin = new(typeof(Range), 0, "begin", IsScalar() & (IsIntegral() | IsFloat()));

    /// <summary>
    /// Gets end.
    /// </summary>
    public static readonly ParameterInfo End = new(typeof(Range), 1, "end", IsScalar() & (IsIntegral() | IsFloat()));

    /// <summary>
    /// Gets step.
    /// </summary>
    public static readonly ParameterInfo Step = new(typeof(Range), 2, "step", IsScalar() & (IsIntegral() | IsFloat()));
}
