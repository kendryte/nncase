// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// ReverseSequence expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ReverseSequence : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ReverseSequence), 0, "input");

    /// <summary>
    /// Gets seq_lens.
    /// </summary>
    public static readonly ParameterInfo SeqLens = new(typeof(ReverseSequence), 1, "seq_lens", IsIntegral() & HasRank(1));

    /// <summary>
    /// Gets batch_axis.
    /// </summary>
    public static readonly ParameterInfo BatchAxis = new(typeof(ReverseSequence), 2, "batch_axis", IsIntegralScalar());

    /// <summary>
    /// Gets time_axis.
    /// </summary>
    public static readonly ParameterInfo TimeAxis = new(typeof(ReverseSequence), 3, "time_axis", IsIntegralScalar());
}
