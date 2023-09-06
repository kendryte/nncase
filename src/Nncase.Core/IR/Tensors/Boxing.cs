// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.Tensors;

/// <summary>
/// Boxing expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Boxing : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Boxing), 0, "input");

    public IRType NewType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{NewType}";
}
