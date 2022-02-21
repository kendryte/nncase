// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// Sigmoid expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Sigmoid() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Sigmoid), 0, "input");
}

/// <summary>
/// Relu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Relu() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Relu), 0, "input");
}

/// <summary>
/// Relu6 expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Relu6() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Relu6), 0, "input");
}

/// <summary>
/// PRelu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record PRelu() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PRelu), 0, "input");
}

/// <summary>
/// LeakyRelu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record LeakyRelu() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(LeakyRelu), 0, "input");
}

/// <summary>
/// Celu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Celu() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Celu), 0, "input");

    /// <summary>
    /// Gets Alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(Celu), 1, "alpha", IsFloatScalar());
}

/// <summary>
/// Selu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Selu() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Selu), 0, "input");

    /// <summary>
    /// Gets Alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(Selu), 1, "alpha", IsFloatScalar());
}

/// <summary>
/// Elu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Elu() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Elu), 0, "input");

    /// <summary>
    /// Gets Alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(Elu), 1, "alpha", IsFloatScalar());
}

/// <summary>
/// HardSwish expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record HardSwish() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(HardSwish), 0, "input");
}

/// <summary>
/// HardSigmoid expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record HardSigmoid() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(HardSigmoid), 0, "input");

    /// <summary>
    /// Gets alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(HardSigmoid), 1, "alpha", IsFloatScalar());

    /// <summary>
    /// Gets beta.
    /// </summary>
    public static readonly ParameterInfo Beta = new(typeof(HardSigmoid), 2, "beta", IsFloatScalar());
}
