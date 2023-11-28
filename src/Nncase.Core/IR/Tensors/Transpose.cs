// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Gets input.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Transpose : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Transpose), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets perm.
    /// </summary>
    public static readonly ParameterInfo Perm = new(typeof(Transpose), 1, "perm", HasRank(1) & IsIntegral());
}
