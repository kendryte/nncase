// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.CPU;

/// <summary>
/// Gather expression.
/// </summary>
public sealed partial class Gather : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Gather), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Index = new(typeof(Gather), 1, "index", IsIntegral(), ParameterKind.Input);

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(Gather), 2, "output");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public int Axis { get; }
}
