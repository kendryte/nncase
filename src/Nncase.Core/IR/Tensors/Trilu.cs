// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Stack expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Trilu : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Trilu), 0, "input");

    /// <summary>
    /// Gets k.
    /// </summary>
    public static readonly ParameterInfo K = new(typeof(Trilu), 1, "k");

    /// <summary>
    /// Gets upper.
    /// </summary>
    public static readonly ParameterInfo Upper = new(typeof(Trilu), 2, "upper");
}
