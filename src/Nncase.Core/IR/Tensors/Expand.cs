// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Expand expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Expand() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Expand), 0, "input");

    /// <summary>
    /// Gets shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(Expand), 1, "shape", HasRank(1));
}
