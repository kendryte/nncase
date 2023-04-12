// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Random;

/// <summary>
/// Normal like expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NormalLike : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NormalLike), 0, "input");

    /// <summary>
    /// Gets mean.
    /// </summary>
    public static readonly ParameterInfo Mean = new(typeof(NormalLike), 1, "mean", IsFloatScalar());

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(NormalLike), 2, "scale", IsFloatScalar());

    /// <summary>
    /// Gets seed.
    /// </summary>
    public static readonly ParameterInfo Seed = new(typeof(NormalLike), 3, "seed", IsFloatScalar());

    public DataType Type { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => Type.GetCSharpName();
}
