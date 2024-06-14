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
/// Gets expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnLSTM : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnLSTM), 0, "input");

    /// <summary>
    /// Gets OutputSize of Ncnn LSTM.
    /// </summary>
    public int OutputSize { get; }

    /// <summary>
    /// Gets HiddenSize of Ncnn LSTM.
    /// </summary>
    public int HiddenSize { get; }

    /// <summary>
    /// Gets WeightDataSize of Ncnn LSTM.
    /// </summary>
    public int WeightDataSize { get; }

    /// <summary>
    /// Gets Direction of Ncnn LSTM.
    /// </summary>
    public int Direction { get; }

    /// <summary>
    /// Gets W of Ncnn LSTM.
    /// </summary>
    public float[] W { get; }

    /// <summary>
    /// Gets B of Ncnn LSTM.
    /// </summary>
    public float[] B { get; }

    /// <summary>
    /// Gets R of Ncnn LSTM.
    /// </summary>
    public float[] R { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"HiddenSize: {HiddenSize}, WeightDataSize: {WeightDataSize}, Direction: {Direction}";
    }
}
