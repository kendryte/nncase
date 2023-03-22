// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Random;

/// <summary>
/// Uniform like expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class UniformLike : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(UniformLike), 0, "input");

    /// <summary>
    /// Gets high.
    /// </summary>
    public static readonly ParameterInfo High = new(typeof(UniformLike), 1, "high", IsFloatScalar());

    /// <summary>
    /// Gets low.
    /// </summary>
    public static readonly ParameterInfo Low = new(typeof(UniformLike), 2, "low", IsFloatScalar());

    /// <summary>
    /// Gets seed.
    /// </summary>
    public static readonly ParameterInfo Seed = new(typeof(UniformLike), 3, "seed", IsFloatScalar());

    public DataType Type { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => Type.GetCSharpName();
}
