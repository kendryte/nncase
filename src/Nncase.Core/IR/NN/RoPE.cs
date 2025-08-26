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
/// RoPE expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class RoPE : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(RoPE), 0, "input", HasRank(3), ParameterKind.Input);

    /// <summary>
    /// Gets cos.
    /// </summary>
    public static readonly ParameterInfo Cos = new(typeof(RoPE), 1, "cos", HasRank(2), ParameterKind.Input);

    /// <summary>
    /// Gets sin.
    /// </summary>
    public static readonly ParameterInfo Sin = new(typeof(RoPE), 2, "sin", HasRank(2), ParameterKind.Input);
}
