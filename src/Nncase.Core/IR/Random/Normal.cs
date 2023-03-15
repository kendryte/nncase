// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Random;

/// <summary>
/// Normal expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Normal : Op
{
    /// <summary>
    /// Gets mean.
    /// </summary>
    public static readonly ParameterInfo Mean = new(typeof(Normal), 0, "mean", IsFloatScalar());

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(Normal), 1, "scale", IsFloatScalar());

    /// <summary>
    /// Gets seed.
    /// </summary>
    public static readonly ParameterInfo Seed = new(typeof(Normal), 2, "seed", IsFloatScalar());

    /// <summary>
    /// Gets shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(Normal), 3, "shape", IsIntegral() & HasRank(1));

    public DataType Type { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => Type.GetCSharpName();
}
