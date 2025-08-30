// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class PackedMatMul : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMul), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMul), 1, "rhs", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public bool FusedReduce { get; }

    public override string DisplayProperty() => $"FusedReduce: {FusedReduce}";
}
