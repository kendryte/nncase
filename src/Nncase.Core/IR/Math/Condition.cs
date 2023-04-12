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
/// Condition operation.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Condition : Op
{
    /// <summary>
    /// Gets Condition.
    /// </summary>
    public static readonly ParameterInfo Predicate = new(typeof(Condition), 0, "predicate", IsBool());

    /// <summary>
    /// Gets Value.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Condition), 1, "value");

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
