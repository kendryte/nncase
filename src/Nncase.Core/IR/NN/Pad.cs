// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// Pad tensor, a little difference with pytorch pad.
/// </summary>
/// <param name="PadMode">Pad mode.</param>
[PatternFunctionalGenerator]
public sealed record Pad(PadMode PadMode) : Op
{
    /// <summary>
    /// input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Pad), 0, "input");

    /// <summary>
    /// `pads` should be a 1D tensor of shape [2 * input_rank]
    /// `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
    /// </summary>
    public static readonly ParameterInfo Pads = new(typeof(Pad), 1, "pads", IsRank(1) & IsIntegral());

    /// <summary>
    /// float pad value.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Pad), 2, "value", IsScalar());
}
