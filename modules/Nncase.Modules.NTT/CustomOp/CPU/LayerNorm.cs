// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.CustomNTT;

[PatternFunctionalGenerator]
public sealed partial class LayerNorm : Op
{
    public static readonly ParameterInfo Input = new(typeof(LayerNorm), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Scale = new(typeof(LayerNorm), 1, "scale", ParameterKind.Input);

    public static readonly ParameterInfo Bias = new(typeof(LayerNorm), 2, "bias", ParameterKind.Input);

    public static readonly ParameterInfo PostScale = new(typeof(LayerNorm), 3, "postScale", IsScalar() | HasShape(new RankedShape(1)), ParameterKind.Attribute);

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public bool ChannelFirst { get; }

    public IRArray<int> VectorizedAxes { get; }

    public IRArray<SBP> InSBPs { get; }

    public IRArray<SBP> ScaleSBPs { get; }

    public IRArray<SBP> BiasSBPs { get; }

    public IRArray<SBP> OutSBPs { get; }

    public Cost Cost { get; }

    public string CSourcePath { get; }

    public string FuncName { get; }

    public DataType OutputDataType { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}, ChannelFirst: {ChannelFirst}";
}
