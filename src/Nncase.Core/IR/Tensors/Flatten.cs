// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.PatternMatch;

namespace Nncase.IR.Tensors;

/// <summary>
/// Flatten expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Flatten : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Flatten), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(Flatten), 1, "axis");
}
