// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Squeeze expression.
/// </summary>
public sealed record Squeeze() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Squeeze), 0, "input");

    /// <summary>
    /// Gets dimension.
    /// </summary>
    public static readonly ParameterInfo Dim = new(typeof(Squeeze), 1, "dim", HasRank(1) & IsIntegral());
}
