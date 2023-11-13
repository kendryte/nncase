// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.Math;

/// <summary>
/// Unary expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Unary : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Unary), 0, "input", ParameterKind.Input);

    public UnaryOp UnaryOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"UnaryOp.{UnaryOp}";
    }
}
