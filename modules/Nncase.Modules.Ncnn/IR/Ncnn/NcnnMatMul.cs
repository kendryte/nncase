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

/// <summary>
/// Matmul expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnMatMul : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo InputA = new(typeof(NcnnMatMul), 0, "inputA");

    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo InputB = new(typeof(NcnnMatMul), 1, "inputB");

    /// <summary>
    /// Gets the flag of which input is const.
    /// </summary>
    public int LorR { get; }

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
        return $"constantShape:{string.Join(",", ConstShape ?? Array.Empty<int>())}";
    }
}
