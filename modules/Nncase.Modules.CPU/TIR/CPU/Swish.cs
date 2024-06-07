// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.CPU;

/// <summary>
/// Swish expression.
/// </summary>
public sealed partial class Swish : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Swish), 0, "input");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Swish), 1, "output");

    /// <summary>
    /// Gets begins.
    /// </summary>
    public float Beta { get; }

    public override string DisplayProperty() => $"Beta: {Beta}";
}
