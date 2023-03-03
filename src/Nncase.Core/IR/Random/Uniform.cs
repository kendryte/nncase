// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Random;

/// <summary>
/// Uniform expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Uniform : Op
{
    /// <summary>
    /// Gets high.
    /// </summary>
    public static readonly ParameterInfo High = new(typeof(Uniform), 0, "high", IsFloatScalar());

    /// <summary>
    /// Gets low.
    /// </summary>
    public static readonly ParameterInfo Low = new(typeof(Uniform), 1, "low", IsFloatScalar());

    /// <summary>
    /// Gets seed.
    /// </summary>
    public static readonly ParameterInfo Seed = new(typeof(Uniform), 2, "seed", IsFloatScalar());

    /// <summary>
    /// Gets shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(Uniform), 3, "shape", IsIntegral() & HasRank(1));

    public DataType Type { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => Type.GetCSharpName();
}
