// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.ArgsStruct;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

/// <summary>
/// Reduction expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnReduction : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnReduction), 0, "input");

    /// <summary>
    /// Gets ReductionArgs of Ncnn Reduction.
    /// </summary>
    public ReductionArgs Args { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"opType:{Args.OpType}, reduceAll:{Args.ReduceAll}, coeff:{Args.Coeff}, axes:{string.Join(",", Args.Axes)}, keepdims:{Args.Keepdims}";
    }
}
