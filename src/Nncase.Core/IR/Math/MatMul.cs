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

public record MatMulDimInfo(int Lm, int Lk, int Rk, int Rn)
{
}

/// <summary>
/// MatMul expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class MatMul : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(MatMul), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(MatMul), 1, "rhs", ParameterKind.Input);

    /// <summary>
    /// Gets TransposeLhs.
    /// </summary>
    public DataType OutputDataType { get; }
}
