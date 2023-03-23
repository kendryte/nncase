// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// LSTM expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class LSTM : Op
{
    /// <summary>
    /// Gets x.
    /// </summary>
    public static readonly ParameterInfo X = new(typeof(LSTM), 0, "x");

    /// <summary>
    /// Gets w.
    /// </summary>
    public static readonly ParameterInfo W = new(typeof(LSTM), 1, "w");

    /// <summary>
    /// Gets r.
    /// </summary>
    public static readonly ParameterInfo R = new(typeof(LSTM), 2, "r");

    /// <summary>
    /// Gets b.
    /// </summary>
    public static readonly ParameterInfo B = new(typeof(LSTM), 3, "b");

    /// <summary>
    /// Gets sequence_lens.
    /// </summary>
    public static readonly ParameterInfo SequenceLens = new(typeof(LSTM), 4, "sequence_lens");

    /// <summary>
    /// Gets initial_h.
    /// </summary>
    public static readonly ParameterInfo InitialH = new(typeof(LSTM), 5, "initial_h");

    /// <summary>
    /// Gets initial_c.
    /// </summary>
    public static readonly ParameterInfo InitialC = new(typeof(LSTM), 6, "initial_c");

    /// <summary>
    /// Gets p.
    /// </summary>
    public static readonly ParameterInfo P = new(typeof(LSTM), 7, "p");

    /// <summary>
    /// Gets activation_alpha.
    /// </summary>
    public static readonly ParameterInfo ActivationAlpha = new(typeof(LSTM), 8, "activation_alpha");

    /// <summary>
    /// Gets activation_beta.
    /// </summary>
    public static readonly ParameterInfo ActivationBeta = new(typeof(LSTM), 9, "activation_beta");

    /// <summary>
    /// Gets clip.
    /// </summary>
    public static readonly ParameterInfo Clip = new(typeof(LSTM), 10, "clip");

    /// <summary>
    /// Gets hidden_size.
    /// </summary>
    public static readonly ParameterInfo HiddenSize = new(typeof(LSTM), 11, "hidden_size");

    /// <summary>
    /// Gets input_forget.
    /// </summary>
    public static readonly ParameterInfo InputForget = new(typeof(LSTM), 12, "input_forget");

    /// <summary>
    /// Gets output_size.
    /// </summary>
    public static readonly ParameterInfo OutputSize = new(typeof(LSTM), 13, "output_size");

    public LSTMDirection Direction { get; }

    public LSTMLayout Layout { get; }

    public string[] Activations { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"LSTMDirection.{Direction}, LSTMLayout.{Layout}, {string.Join(", ", Activations.Select(s => "\"" + s + "\""))}";
}
