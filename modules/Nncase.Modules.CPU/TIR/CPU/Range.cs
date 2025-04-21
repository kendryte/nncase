// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Range : CPUKernelOp
{
    /// <summary>
    /// Gets begin.
    /// </summary>
    public static readonly ParameterInfo Begin = new(typeof(Range), 0, "begin");

    /// <summary>
    /// Gets end.
    /// </summary>
    public static readonly ParameterInfo End = new(typeof(Range), 1, "end");

    /// <summary>
    /// Gets step.
    /// </summary>
    public static readonly ParameterInfo Step = new(typeof(Range), 2, "step");

}
