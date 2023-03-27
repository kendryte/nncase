// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// Conv2DTranspose.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Conv2DTranspose : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Conv2DTranspose), 0, "input");

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(Conv2DTranspose), 1, "weights");

    /// <summary>
    /// Gets Bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(Conv2DTranspose), 2, "bias");

    /// <summary>
    /// Gets OutputShape.
    /// </summary>
    public static readonly ParameterInfo OutputShape = new(typeof(Conv2DTranspose), 3, "outputShape");

    /// <summary>
    /// Gets Stride.
    /// </summary>
    public static readonly ParameterInfo Stride = new(typeof(Conv2DTranspose), 4, "stride");

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public static readonly ParameterInfo Padding = new(typeof(Conv2DTranspose), 5, "padding");

    /// <summary>
    /// Gets Output Padding.
    /// </summary>
    public static readonly ParameterInfo OutputPadding = new(typeof(Conv2DTranspose), 6, "output_padding");

    /// <summary>
    /// Gets Dilation.
    /// </summary>
    public static readonly ParameterInfo Dilation = new(typeof(Conv2DTranspose), 7, "dilation");

    /// <summary>
    /// Gets Groups.
    /// </summary>
    public static readonly ParameterInfo Groups = new(typeof(Conv2DTranspose), 8, "groups");

    /// <summary>
    /// Gets FusedClamp.
    /// </summary>
    public static readonly ParameterInfo FusedClamp = new(typeof(Conv2DTranspose), 9, "fused_clamp", HasShape(new Shape(2)) & HasDataType(DataTypes.Float32));

    public PadMode PadMode { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"PadMode.{PadMode}";
}
