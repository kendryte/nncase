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
/// ScatterND expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ScatterND : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ScatterND), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets indices.
    /// </summary>
    public static readonly ParameterInfo Indices = new(typeof(ScatterND), 1, "indices");

    /// <summary>
    /// Gets updates.
    /// </summary>
    public static readonly ParameterInfo Updates = new(typeof(ScatterND), 2, "updates");
}
