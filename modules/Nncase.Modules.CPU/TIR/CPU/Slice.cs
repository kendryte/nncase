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
/// Slice expression.
/// </summary>
public sealed partial class Slice : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Slice), 0, "input");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Slice), 1, "output");

    /// <summary>
    /// Gets begins.
    /// </summary>
    public IRArray<int> Begins { get; }

    /// <summary>
    /// Gets ends.
    /// </summary>
    public IRArray<int> Ends { get; }

    /// <summary>
    /// Gets axes.
    /// </summary>
    public IRArray<int> Axes { get; }

    /// <summary>
    /// Gets strides.
    /// </summary>
    public IRArray<int> Strides { get; }

    public DistributedType DistType { get; }
}
