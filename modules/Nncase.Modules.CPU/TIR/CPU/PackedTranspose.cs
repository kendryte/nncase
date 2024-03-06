// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.CPU;

[PatternFunctionalGenerator]
public sealed partial class PackedTranspose : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PackedTranspose), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets perm.
    /// </summary>
    public static readonly ParameterInfo Perm = new(typeof(PackedTranspose), 1, "perm", HasRank(1) & IsIntegral());

    public static readonly ParameterInfo Output = new(typeof(PackedTranspose), 2, "output", ParameterKind.Input);

    public IRArray<int> PackedAxes { get; }
}
