// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.K210;

/// <summary>
/// KPU Conv2D.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial record class KPUConv2D(bool IsDepthwise, KPUFilterType FilterType, KPUPoolType PoolType, KPUActivationParameters Activation) : Op
{
    /*public record struct KPUConv2dQuantArgs
    {
        public Int32 argX{ get; set; }
        public Int32 shiftX{ get; set; }
        public Int32 argW{ get; set; }
        public Int32 shiftW{ get; set; }
        public Int64 argAdd{ get; set; }
    }*/

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(KPUConv2D), 0, "input", HasRank(4));

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(KPUConv2D), 1, "weights", HasRank(4));

    /// <summary>
    /// Gets batch norms.
    /// </summary>
    public static readonly ParameterInfo BatchNorms = new(typeof(KPUConv2D), 2, "batchNorms", HasRank(1));

    /// <summary>
    /// Gets activation.
    /// </summary>
    public static readonly ParameterInfo OutputQuantParam = new(typeof(KPUConv2D), 3, "outputQuantParam",HasRank(4));

    /// <summary>
    /// Gets FusedClamp.
    /// </summary>
    public static readonly ParameterInfo FusedClamp = new(typeof(KPUConv2D), 4, "fused_clamp", HasShape(new Shape(2)) & HasDataType(DataTypes.BFloat16));

    /// <summary>
    /// Gets IsDepthwise.
    /// </summary>
    // public static readonly ParameterInfo IsDepthwise = new(typeof(KPUConv2D), 5, "is_depthwise");

    /// <summary>
    /// Gets FilterType.
    /// </summary>
    // public static readonly ParameterInfo FilterType = new(typeof(KPUConv2D), 6, "filter_type");

    /// <summary>
    /// Gets PadValue.
    /// </summary>
    public static readonly ParameterInfo PadValue = new(typeof(KPUConv2D), 7, "pad_value");

    /// <summary>
    /// Gets PoolType.
    /// </summary>
    // public static readonly ParameterInfo PoolType = new(typeof(KPUConv2D), 8, "pool_type");

    /// <summary>
    /// Gets QuantArgs.
    /// </summary>
    public static readonly ParameterInfo QuantArgs = new(typeof(KPUConv2D), 9, "quant_args");
    /*/// <summary>
    /// Gets argX
    /// </summary>
    public static readonly ParameterInfo ArgX = new(typeof(KPUConv2D), 4, "argX",HasRank(2));

    /// <summary>
    /// Gets shiftX
    /// </summary>
    public static readonly ParameterInfo ShiftX = new(typeof(KPUConv2D), 5, "shiftX",IsIntegral());

    /// <summary>
    /// Gets argW
    /// </summary>
    public static readonly ParameterInfo ArgW = new(typeof(KPUConv2D), 6, "argW",IsIntegral());

    /// <summary>
    /// Gets shiftW
    /// </summary>
    public static readonly ParameterInfo ShiftW = new(typeof(KPUConv2D), 7, "shiftW",IsIntegral());

    /// <summary>
    /// Gets argAdd
    /// </summary>
    public static readonly ParameterInfo ArgAdd = new(typeof(KPUConv2D), 8, "argAdd",IsIntegral());

    /*
    /// <summary>
    /// Gets Padding
    /// </summary>
    public static readonly ParameterInfo Padding = new(typeof(KPUConv2D), 9, "Padding",IsIntegral());
    */

}
