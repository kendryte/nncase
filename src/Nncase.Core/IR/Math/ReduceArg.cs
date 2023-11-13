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

namespace Nncase.IR.Math;

/// <summary>
/// ReduceArg expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ReduceArg : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ReduceArg), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets Axis.
    /// </summary>
    /// <remarks>Named dim in torch.</remarks>
    public static readonly ParameterInfo Axis = new(typeof(ReduceArg), 1, "axis", IsIntegralScalar());

    /// <summary>
    /// Gets whether to keep dims.
    /// </summary>
    public static readonly ParameterInfo KeepDims = new(typeof(ReduceArg), 2, "keepDims", IsBoolScalar());

    /// <summary>
    /// Gets whether to select the last index.
    /// </summary>
    /// <remarks>Only used in onnx.</remarks>
    public static readonly ParameterInfo SelectLastIndex = new(typeof(ReduceArg), 3, "selectLastIndex", IsBoolScalar());

    public ReduceArgOp ReduceArgOp { get; }

    public PrimType DestType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"ReduceArgOp.{ReduceArgOp}, DestType: {DestType}";
}
