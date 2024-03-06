// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

/// <summary>
/// Pack expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Pack : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Pack), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(Pack), 1, "output", ParameterKind.Input);

    public IRArray<int> Lanes { get; }

    public IRArray<int> Axes { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"Lanes: {Lanes}, Axes: {Axes}";
}
