// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

/// <summary>
/// Pack expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Pack : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Pack), 0, "input");

    public int Lanes { get; }

    public int Axis { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{Lanes}, {Axis}";
}
