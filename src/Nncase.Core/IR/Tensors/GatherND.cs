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
public sealed partial class GatherND : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(GatherND), 0, "input");

    /// <summary>
    /// Gets batch dims.
    /// </summary>
    public static readonly ParameterInfo BatchDims = new(typeof(GatherND), 1, "batch_dims");

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Index = new(typeof(GatherND), 2, "index");
}
