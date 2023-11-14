// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

[PatternFunctionalGenerator]
public sealed partial class L2Normalization : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");
}

[PatternFunctionalGenerator]
public sealed partial class BatchNormalization : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BatchNormalization), 0, "input");

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(BatchNormalization), 1, "scale", HasRank(1));

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(BatchNormalization), 2, "bias", HasRank(1));

    /// <summary>
    /// Gets input_mean.
    /// </summary>
    public static readonly ParameterInfo InputMean = new(typeof(BatchNormalization), 3, "input_mean", HasRank(1));

    /// <summary>
    /// Gets input_var.
    /// </summary>
    public static readonly ParameterInfo InputVar = new(typeof(BatchNormalization), 4, "input_var", HasRank(1));

    /// <summary>
    /// Gets epsilon.
    /// </summary>
    public static readonly ParameterInfo Epsilon = new(typeof(BatchNormalization), 5, "epsilon", IsFloatScalar());

    /// <summary>
    /// Gets momentum.
    /// </summary>
    public static readonly ParameterInfo Momentum = new(typeof(BatchNormalization), 6, "momentum", IsFloatScalar());
}

[PatternFunctionalGenerator]
public sealed partial class InstanceNormalization : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(InstanceNormalization), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(InstanceNormalization), 1, "scale", ParameterKind.Input);

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(InstanceNormalization), 2, "bias", ParameterKind.Input);

    /// <summary>
    /// Gets Epsilon.
    /// </summary>
    public static readonly ParameterInfo Epsilon = new(typeof(InstanceNormalization), 3, "epsilon", IsFloatScalar());
}

[PatternFunctionalGenerator]
public sealed partial class LpNormalization : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(LpNormalization), 0, "input", IsFloat());

    /// <summary>
    /// Gets Axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(LpNormalization), 1, "axis", IsIntegralScalar());

    /// <summary>
    /// Gets P.
    /// </summary>
    public static readonly ParameterInfo P = new(typeof(LpNormalization), 2, "p", IsFloatScalar());
}

[PatternFunctionalGenerator]
public sealed partial class LRN : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(LRN), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Alpha = new(typeof(LRN), 1, "alpha", IsFloatScalar());

    /// <summary>
    /// Gets beta.
    /// </summary>
    public static readonly ParameterInfo Beta = new(typeof(LRN), 2, "beta", IsFloatScalar());

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(LRN), 3, "bias", IsFloatScalar());

    /// <summary>
    /// Gets size.
    /// </summary>
    public static readonly ParameterInfo Size = new(typeof(LRN), 4, "size", IsIntegralScalar());
}
