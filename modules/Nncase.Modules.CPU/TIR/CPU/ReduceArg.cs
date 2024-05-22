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
/// ReduceArg expression.
/// </summary>
public sealed partial class ReduceArg : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ReduceArg), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(ReduceArg), 1, "output");

    /// <summary>
    /// Gets Axis.
    /// </summary>
    /// <remarks>Named dim in torch.</remarks>
    public int Axis { get; }

    /// <summary>
    /// Gets a value indicating whether gets whether to keep dims.
    /// </summary>
    public bool KeepDims { get; }

    /// <summary>
    /// Gets a value indicating whether gets whether to select the last index.
    /// </summary>
    /// <remarks>Only used in onnx.</remarks>
    public bool SelectLastIndex { get; }

    public ReduceArgOp ReduceArgOp { get; }

    public DataType DestType { get; }
}
