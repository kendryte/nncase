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

namespace Nncase.TIR.NTT;

/// <summary>
/// Stack expression.
/// </summary>
public sealed partial class Stack : NTTKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Stack), 0, "input");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Stack), 1, "output");

    /// <summary>
    /// Gets begins.
    /// </summary>
    public int Axis { get; }
}
