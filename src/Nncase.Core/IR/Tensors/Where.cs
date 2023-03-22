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
/// Where expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Where : Op
{
    /// <summary>
    /// Gets condition.
    /// </summary>
    public static readonly ParameterInfo Cond = new(typeof(Where), 0, "cond");

    /// <summary>
    /// Gets x.
    /// </summary>
    public static readonly ParameterInfo X = new(typeof(Where), 1, "x");

    /// <summary>
    /// Gets y.
    /// </summary>
    public static readonly ParameterInfo Y = new(typeof(Where), 2, "y");

    public bool IsTfWhere { get; }
}
