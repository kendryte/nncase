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

namespace Nncase.TIR.CPU;

/// <summary>
/// Pack expression.
/// </summary>
public sealed partial class Conv2D : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Conv2D), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Weights = new(typeof(Conv2D), 1, "weights", ParameterKind.Input);

    public static readonly ParameterInfo Bias = new(typeof(Conv2D), 2, "bias", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(Conv2D), 3, "output", ParameterKind.Input);

    /// <summary>
    /// Gets Stride.
    /// </summary>
    public IRArray<int> Stride { get; }

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public IRArray<int> Padding { get; }

    /// <summary>
    /// Gets Dilation.
    /// </summary>
    public IRArray<int> Dilation { get; }

    /// <summary>
    /// Gets Groups.
    /// </summary>
    public int Groups { get; }

    public PadMode PadMode { get; }
}
