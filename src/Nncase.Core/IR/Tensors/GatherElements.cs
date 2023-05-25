// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.PatternMatch;

namespace Nncase.IR.Tensors;

/// <summary>
/// GatherND expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GatherElements : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(GatherElements), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(GatherElements), 1, "axis");

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Indices = new(typeof(GatherElements), 2, "indices");
}
