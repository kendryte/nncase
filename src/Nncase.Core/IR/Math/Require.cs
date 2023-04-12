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
/// Require expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Require : Op
{
    /// <summary>
    /// Gets Condition.
    /// </summary>
    public static readonly ParameterInfo Predicate = new(typeof(Require), 0, "predicate", IsBool());

    /// <summary>
    /// Gets FalseValue.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Require), 1, "value");

    public string Message { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => "\"\"";
}
