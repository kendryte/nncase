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
/// Binary expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Binary : Op
{
    /// <summary>
    /// Gets lhs.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets rhs.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "rhs", ParameterKind.Input);

    public BinaryOp BinaryOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"BinaryOp.{BinaryOp}";
    }
}
