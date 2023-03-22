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

namespace Nncase.IR.Math;

/// <summary>
/// CumSum expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class CumSum : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(CumSum), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(CumSum), 1, "axis");

    /// <summary>
    /// Gets exclusive.
    /// </summary>
    public static readonly ParameterInfo Exclusive = new(typeof(CumSum), 2, "exclusive", IsBoolScalar());

    /// <summary>
    /// Gets reverse.
    /// </summary>
    public static readonly ParameterInfo Reverse = new(typeof(CumSum), 3, "reverse", IsBoolScalar());
}
