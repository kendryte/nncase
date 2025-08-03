// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NN;

[PatternFunctionalGenerator]
public sealed partial class MLAPagedAttention : Op
{
    public static readonly ParameterInfo Q = new(typeof(MLAPagedAttention), 0, "q", ParameterKind.Input);
    public static readonly ParameterInfo KVCaches = new(typeof(MLAPagedAttention), 1, "kvCaches", ParameterKind.Attribute);
    public static readonly ParameterInfo Extra = new(typeof(MLAPagedAttention), 2, "extra", ParameterKind.Input);
    public static readonly ParameterInfo Scale = new(typeof(MLAPagedAttention), 3, "scale", ParameterKind.Attribute);
    public static readonly ParameterInfo QAProj = new(typeof(MLAPagedAttention), 4, "qaProj", ParameterKind.Attribute);
    public static readonly ParameterInfo QAProjScale = new(typeof(MLAPagedAttention), 5, "qaProjScale", ParameterKind.Attribute);
    public static readonly ParameterInfo QALayerNormW = new(typeof(MLAPagedAttention), 6, "qaLayerNormW", ParameterKind.Attribute);
    public static readonly ParameterInfo QBProj = new(typeof(MLAPagedAttention), 7, "qbProj", ParameterKind.Attribute);
    public static readonly ParameterInfo QBProjScale = new(typeof(MLAPagedAttention), 8, "qbProjScale", ParameterKind.Attribute);
    public static readonly ParameterInfo KVALayerNormW = new(typeof(MLAPagedAttention), 9, "kvaLayerNormW", ParameterKind.Attribute);
    public static readonly ParameterInfo KVBProj = new(typeof(MLAPagedAttention), 10, "kvbProj", ParameterKind.Attribute);
    public static readonly ParameterInfo KVBProjScale = new(typeof(MLAPagedAttention), 11, "kvbProjScale", ParameterKind.Attribute);

    public int LayerId { get; }

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }

    public int NumAttentionHeads { get; }

    public int KVLoraRank { get; }

    public int QKNopeHeadDim { get; }

    public int QKRopeHeadDim { get; }

    public int VHeadDim { get; }

    public override string DisplayProperty() => $"LayerId: {LayerId}, Layout [{string.Join(',', Layout)}], HiddenSize: {HiddenSize}, NumAttentionHeads: {NumAttentionHeads}, KVLoraRank: {KVLoraRank}, QKNopeHeadDim: {QKNopeHeadDim}, QKRopeHeadDim: {QKRopeHeadDim}, VHeadDim: {VHeadDim}";
}
