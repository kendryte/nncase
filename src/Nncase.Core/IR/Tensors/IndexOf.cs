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
/// Prod expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class IndexOf : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(IndexOf), 0, "input", HasRank(1) & IsIntegral());

    /// <summary>
    /// Value.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(IndexOf), 1, "value", IsScalar() & IsIntegral());
}
