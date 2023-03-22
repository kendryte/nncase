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
/// The base class.
/// </summary>
public abstract class ActivationOp : Op
{
}

/// <summary>
/// Sigmoid expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Sigmoid : ActivationOp
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
public sealed partial class Relu : ActivationOp
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
public sealed partial class Relu6 : ActivationOp
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
public sealed partial class PRelu : ActivationOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PRelu), 0, "input");

    /// <summary>
    /// Gets Slope.
    /// </summary>
    public static readonly ParameterInfo Slope = new(typeof(PRelu), 1, "slope");
}

/// <summary>
/// LeakyRelu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class LeakyRelu : ActivationOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(LeakyRelu), 0, "input");

    /// <summary>
    /// Gets Alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(LeakyRelu), 1, "alpha");
}

/// <summary>
/// Celu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Celu : ActivationOp
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
public sealed partial class Selu : ActivationOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Selu), 0, "input");

    /// <summary>
    /// Gets Alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(Selu), 1, "alpha", IsFloatScalar());

    /// <summary>
    /// Gets Gamma.
    /// </summary>
    public static readonly ParameterInfo Gamma = new(typeof(Selu), 2, "gamma", IsFloatScalar());
}

/// <summary>
/// Elu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Elu : ActivationOp
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
/// Swish expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Swish : ActivationOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Swish), 0, "input");
}

/// <summary>
/// HardSwish expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class HardSwish : ActivationOp
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
public sealed partial class HardSigmoid : ActivationOp
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

/// <summary>
/// Erf expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Erf : ActivationOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Erf), 0, "input");
}

/// <summary>
/// Gelu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Gelu : ActivationOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Gelu), 0, "input");

    /// <summary>
    /// Gets alpha.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(Gelu), 1, "alpha", IsFloatScalar());
}
