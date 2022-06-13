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
public sealed record class KPUConv2D(bool IsDepthwise, KPUFilterType FilterType, KPUPoolType PoolType, Tensor<KPUBatchNormSegment> BatchNorms, KPUActivationParameters Activation) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(FakeKPUConv2D), 0, "input", HasRank(4));

    /// <summary>
    /// Gets Weights.
    /// </summary>
    public static readonly ParameterInfo Weights = new(typeof(FakeKPUConv2D), 1, "weights", HasRank(4));
}
