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
/// Concat expression.
/// </summary>
public sealed partial class Concat : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Concat), 0, "input");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Concat), 1, "output");

    /// <summary>
    /// Gets begins.
    /// </summary>
    public int Axis { get; }
}
