// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// ReduceWindow2D.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ReduceWindow2D : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ReduceWindow2D), 0, "input");

    /// <summary>
    /// Get initial value.
    /// </summary>
    public static readonly ParameterInfo InitValue = new(typeof(ReduceWindow2D), 1, "initValue", IsScalar());

    /// <summary>
    /// Get filter.
    /// </summary>
    public static readonly ParameterInfo Filter = new(typeof(ReduceWindow2D), 2, "filter", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets Stride.
    /// </summary>
    public static readonly ParameterInfo Stride = new(typeof(ReduceWindow2D), 3, "stride", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public static readonly ParameterInfo Padding = new(typeof(ReduceWindow2D), 4, "padding", HasRank(2) & IsIntegral());

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public static readonly ParameterInfo Dilation = new(typeof(ReduceWindow2D), 5, "dilation", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets CeilMode.
    /// </summary>
    public static readonly ParameterInfo CeilMode = new(typeof(ReduceWindow2D), 6, "ceilMode", IsBool());

    /// <summary>
    /// Gets CountIncludePad.
    /// </summary>
    public static readonly ParameterInfo CountIncludePad = new(typeof(ReduceWindow2D), 7, "countIncludePad", IsBool());

    public ReduceOp ReduceOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"ReduceOp.{ReduceOp}";
}
