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

namespace Nncase.IR.NN;

/// <summary>
/// Conv2D.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Conv2D : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Conv2D), 0, "input");

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(Conv2D), 1, "weights", HasRank(4));

    /// <summary>
    /// Gets Bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(Conv2D), 2, "bias", HasRank(1));

    /// <summary>
    /// Gets Stride.
    /// </summary>
    public static readonly ParameterInfo Stride = new(typeof(Conv2D), 3, "stride", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets Padding.
    /// </summary>
    public static readonly ParameterInfo Padding = new(typeof(Conv2D), 4, "padding", HasRank(2) & IsIntegral());

    /// <summary>
    /// Gets Dilation.
    /// </summary>
    public static readonly ParameterInfo Dilation = new(typeof(Conv2D), 5, "dilation", HasRank(1) & IsIntegral());

    /// <summary>
    /// Gets Groups.
    /// </summary>
    public static readonly ParameterInfo Groups = new(typeof(Conv2D), 6, "groups", IsScalar() & IsIntegral());

    /// <summary>
    /// Gets FusedClamp.
    /// </summary>
    public static readonly ParameterInfo FusedClamp = new(typeof(Conv2D), 7, "fused_clamp", HasShape(new Shape(2)) & HasDataType(DataTypes.Float32));

    public PadMode PadMode { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"PadMode.{PadMode}";
}
