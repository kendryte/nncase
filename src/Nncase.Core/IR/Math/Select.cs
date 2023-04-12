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
/// Unary expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Select : Op
{
    /// <summary>
    /// Gets Condition.
    /// </summary>
    public static readonly ParameterInfo Predicate = new(typeof(Select), 0, "predicate", IsBool());

    /// <summary>
    /// Gets TrueValue.
    /// </summary>
    public static readonly ParameterInfo TrueValue = new(typeof(Select), 1, "true_value");

    /// <summary>
    /// Gets FalseValue.
    /// </summary>
    public static readonly ParameterInfo FalseValue = new(typeof(Select), 2, "false_value");
}
