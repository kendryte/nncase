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

public enum BinaryOperationType
{
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    MAX = 4,
    MIN = 5,
    POW = 6,

    // Not support below
    RSUB = 7,
    RDIV = 8,
    RPOW = 9,
    ATAN2 = 10,
    RATAN2 = 11,
}

/// <summary>
/// Binary expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnBinary : Op
{
    /// <summary>
    /// Gets inputA.
    /// </summary>
    public static readonly ParameterInfo InputA = new(typeof(NcnnBinary), 0, "inputA");

    /// <summary>
    /// Gets inputB.
    /// </summary>
    public static readonly ParameterInfo InputB = new(typeof(NcnnBinary), 1, "inputB");

    public BinaryOperationType OpType { get; }

    /// <summary>
    /// Gets the flag of which input is const.
    /// </summary>
    public int LorR { get; }

    // These args will never used in nncase, scaler input was convert to single constant.
    // So, it was no longer be scaler.
    // public int WithScaler { get; }
    // public float B { get; }

    /// <summary>
    /// Gets constant data.
    /// </summary>
    public float[] ConstInput { get; }

    /// <summary>
    /// Gets shape of constant data.
    /// </summary>
    public int[] ConstShape { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"BinaryOp.{OpType}";
    }
}
