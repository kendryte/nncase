// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Math;

/// <summary>
/// Reduce expression.
/// </summary>
public sealed record Reduce(ReduceOp ReduceOp) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Reduce), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(Reduce), 1, "axis", IsIntegral() & IsRank(1));

    /// <summary>
    /// Gets initial value.
    /// </summary>
    public static readonly ParameterInfo InitValue = new(typeof(Reduce), 2, "initValue", IsScalar());

    /// <summary>
    /// Gets whether to keep dims.
    /// </summary>
    public static readonly ParameterInfo KeepDims = new(typeof(Reduce), 3, "keepDims", IsScalar() & IsIntegral());
}
