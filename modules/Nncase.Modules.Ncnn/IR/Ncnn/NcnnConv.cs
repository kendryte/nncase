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
    public static readonly ParameterInfo Input = new(typeof(NcnnConv), 0, "input");

    /// <summary>
    /// Gets input.
    /// </summary>
    public float[] WeightData { get; }

    /// <summary>
    /// Gets BiasData.
    /// </summary>
    public float[] BiasData { get; }

    /// <summary>
    /// Gets NumOutput.
    /// </summary>
    public int NumOutput { get; }

    /// <summary>
    /// Gets KernelW.
    /// </summary>
    public int KernelW { get; }

    /// <summary>
    /// Gets KernelH.
    /// </summary>
    public int KernelH { get; }

    /// <summary>
    /// Gets DilationW.
    /// </summary>
    public int DilationW { get; }

    /// <summary>
    /// Gets DilationH.
    /// </summary>
    public int DilationH { get; }

    /// <summary>
    /// Gets StrideW.
    /// </summary>
    public int StrideW { get; }

    /// <summary>
    /// Gets StrideH.
    /// </summary>
    public int StrideH { get; }

    /// <summary>
    /// Gets PadLeft.
    /// </summary>
    public int PadLeft { get; }

    /// <summary>
    /// Gets PadRight.
    /// </summary>
    public int PadRight { get; }

    /// <summary>
    /// Gets PadTop.
    /// </summary>
    public int PadTop { get; }

    /// <summary>
    /// Gets PadBottom.
    /// </summary>
    public int PadBottom { get; }

    /// <summary>
    /// Gets PadValue.
    /// </summary>
    public float PadValue { get; }

    /// <summary>
    /// Gets BiasTerm.
    /// </summary>
    public int BiasTerm { get; }

    /// <summary>
    /// Gets WeightDataSize.
    /// </summary>
    public int WeightDataSize { get; }

    /// <summary>
    /// Gets Int8ScaleTerm.
    /// </summary>
    public int Int8ScaleTerm { get; }

    /// <summary>
    /// Gets ActivationType.
    /// </summary>
    public int ActivationType { get; }

    /// <summary>
    /// Gets ActivationParams.
    /// </summary>
    public float[] ActivationParams { get; }

    /// <summary>
    /// Gets DynamicWeight.
    /// </summary>
    public int DynamicWeight { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"";
    }
}
