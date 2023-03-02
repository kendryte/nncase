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
/// Hardmax expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record LayerNorm(int Axis, float Epsilon) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Hardmax), 0, "input");

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(Hardmax), 1, "scale");

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(Hardmax), 2, "bias");
}
