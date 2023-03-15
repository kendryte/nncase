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
public sealed partial class TopK : Op
{
    /// <summary>
    /// Gets x.
    /// </summary>
    public static readonly ParameterInfo X = new(typeof(TopK), 0, "x");

    /// <summary>
    /// Gets k.
    /// </summary>
    public static readonly ParameterInfo K = new(typeof(TopK), 1, "k");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(TopK), 2, "axis");

    /// <summary>
    /// Gets largest.
    /// </summary>
    public static readonly ParameterInfo Largest = new(typeof(TopK), 3, "largest");

    /// <summary>
    /// Gets sorted.
    /// </summary>
    public static readonly ParameterInfo Sorted = new(typeof(TopK), 4, "sorted");
}
