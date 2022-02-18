// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Pattern;
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
    /// pads , shape is [channels, 2], eg. [[1,1],
    ///                                     [2,2]]  mean pad shape [1,2,3,4] =>  [1,2,5,8].
    /// </summary>
    public static readonly ParameterInfo Pads = new(typeof(Pad), 1, "pads", IsRank(2) & IsIntegral());

    /// <summary>
    /// float pad value.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Pad), 2, "value", IsScalar());
}
