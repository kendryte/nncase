// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

/// <summary>
/// Conv expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnConv : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo InputA = new(typeof(NcnnConv), 0, "input");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(NcnnConv), 1, "weights");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(NcnnConv), 2, "bias");

    public int NumOutput { get; }
    public int KernelW { get; }
    public int KernelH { get; }
    public int DilationW { get; }
    public int DilationH { get; }
    public int StrideW { get; }
    public int StrideH { get; }
    public int PadLeft { get; }
    public int PadRight { get; }
    public int PadTop { get; }
    public int PadBotton { get; }
    public float PadValue { get; }
    public int BiasTerm { get; }
    public int WeightDataSize { get; }
    // public int Int8ScaleTerm { get; }
    public int ActivationType { get; }
    public float[] ActivationParams { get; }
    public int DynamicWeight { get; }

    /// <summary>
    /// Gets constant data.
    /// </summary>
    public float[] Weight { get; }
    public int[] Weight { get; }

    /// <summary>
    /// Gets shape of constant data.
    /// </summary>
    public IR.Tensor ConstShape { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"ConvOp.{OpType}";
    }
}
