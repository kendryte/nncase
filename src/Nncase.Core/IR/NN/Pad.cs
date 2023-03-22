// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// Pad tensor, a little difference with pytorch pad.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Pad : Op
{
    /// <summary>
    /// input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Pad), 0, "input");

    /// <summary>
    /// [1, 2, 3, 4] [[0, 0, 0, 0, 1, 1, 2, 2]] ⇒ [1, 2, 5, 8].
    /// </summary>
    public static readonly ParameterInfo Pads = new(typeof(Pad), 1, "pads", HasRank(1) & IsIntegral());

    /// <summary>
    /// float pad value.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Pad), 2, "value", IsScalar());

    /// <summary>
    /// Gets pad mode.
    /// </summary>
    public PadMode PadMode { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"PadMode.{PadMode}";
}
