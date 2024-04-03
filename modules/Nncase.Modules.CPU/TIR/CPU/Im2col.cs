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
/// Im2col expression.
/// </summary>
public sealed partial class Im2col : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Im2col), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(Im2col), 1, "output", ParameterKind.Input);

    public IRArray<int> Kernel { get; }

    public IRArray<int> Stride { get; }

    public IRArray<int> Padding { get; }
}
