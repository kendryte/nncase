// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Pattern;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Tensors;

/// <summary>
/// Concat expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Concat() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Concat), 0, "inputs");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(Concat), 1, "axis");
}
