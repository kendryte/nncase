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
/// MatMul expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class QLinearMatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(QLinearMatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(QLinearMatMul), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo LhsScale = new(typeof(QLinearMatMul), 2, "lhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo LhsZeroPoint = new(typeof(QLinearMatMul), 3, "lhs_zero_point", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(QLinearMatMul), 4, "rhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo RhsZeroPoint = new(typeof(QLinearMatMul), 5, "rhs_zero_point", ParameterKind.Input);

    public static readonly ParameterInfo OutputScale = new(typeof(QLinearMatMul), 6, "output_scale", ParameterKind.Input);

    public static readonly ParameterInfo OutputZeroPoint = new(typeof(QLinearMatMul), 7, "output_zero_point", ParameterKind.Input);

    public DataType OutputDataType { get; }
}
