// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

public enum UnaryOperationType
{
    ABS = 0,
    NEG = 1,
    FLOOR = 2,
    CEIL = 3,
    SQUARE = 4,
    SQRT = 5,
    RSQRT = 6,
    EXP = 7,
    LOG = 8,
    SIN = 9,
    COS = 10,
    TAN = 11,
    ASIN = 12,
    ACOS = 13,
    ATAN = 14,
    RECIPROCAL = 15,
    TANH = 16,
    LOG10 = 17,
    ROUND = 18,
    TRUNC = 19,
}

/// <summary>
/// Unary expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnUnary : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnUnary), 0, "input");

    public UnaryOperationType OpType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"UnaryOp.{OpType}";
    }
}
