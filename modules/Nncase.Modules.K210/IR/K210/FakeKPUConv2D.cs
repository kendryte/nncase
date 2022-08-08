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
/// Fake KPU Conv2D.
/// </summary>
[PatternFunctionalGenerator]
public sealed record class FakeKPUConv2D(bool IsDepthwise, KPUFilterType FilterType, KPUPoolType PoolType, Tensor<float> Bias, ValueRange<float> FusedClamp) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(FakeKPUConv2D), 0, "input", HasRank(4));

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(FakeKPUConv2D), 1, "weights", HasRank(4));
    
    /*
    /// <summary>
    /// Gets batch norms.
    /// </summary>
    public static readonly ParameterInfo BatchNorms = new(typeof(KPUConv2D), 2, "batchNorms", ValidFakeBatchNorms());

    /// <summary>
    /// Gets activation.
    /// </summary>
    public static readonly ParameterInfo OutputQuantParam = new(typeof(KPUConv2D), 3, "outputQuantParam",ValidFakeAct());
*/
}
