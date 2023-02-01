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
public sealed partial record class KPUConv2D(bool IsDepthwise, KPUFilterType FilterType, KPUPoolType PoolType,
    KPUActivationParameters Act, KPUBatchNormParameters Bn, Kpu_conv2d_quant_args Quant_args) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(KPUConv2D), 0, "input", HasRank(4) & HasDataType(DataTypes.UInt8));

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(KPUConv2D), 1, "weights", HasRank(4) & HasDataType(DataTypes.UInt8));

    /// <summary>
    /// Gets batch norms.
    /// </summary>
    public static readonly ParameterInfo BatchNorms = new(typeof(KPUConv2D), 2, "batchNorms", HasRank(1) & HasDataType(DataTypes.UInt64));

    /// <summary>
    /// Gets activation.
    /// </summary>
    public static readonly ParameterInfo OutputQuantParam =
        new(typeof(KPUConv2D), 3, "outputQuantParam", HasRank(4) & HasDataType(DataTypes.UInt8));

    /// <summary>
    /// Gets PadValue.
    /// </summary>
    public static readonly ParameterInfo
        PadValue = new(typeof(KPUConv2D), 4, "pad_value", HasDataType(DataTypes.UInt8));
}
