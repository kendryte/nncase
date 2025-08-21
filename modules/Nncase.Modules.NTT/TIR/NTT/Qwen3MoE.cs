// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

public sealed partial class Qwen3MoE : Nncase.TIR.NTT.NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(Qwen3MoE), 0, "q", ParameterKind.Input);
    public static readonly ParameterInfo MoeGateW = new(typeof(Qwen3MoE), 1, "MoeGateW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertGateInputScale = new(typeof(Qwen3MoE), 2, "MoeExpertGateInputScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertGateProjW = new(typeof(Qwen3MoE), 3, "MoeExpertGateProjW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertGateProjScale = new(typeof(Qwen3MoE), 4, "MoeExpertGateProjScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertDownInputScale = new(typeof(Qwen3MoE), 5, "MoeExpertDownInputScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertDownProjW = new(typeof(Qwen3MoE), 6, "MoeExpertDownProjW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertDownProjScale = new(typeof(Qwen3MoE), 7, "MoeExpertDownProjScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertUpInputScale = new(typeof(Qwen3MoE), 8, "MoeExpertUpInputScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertUpProjW = new(typeof(Qwen3MoE), 9, "MoeExpertUpProjW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertUpProjScale = new(typeof(Qwen3MoE), 10, "MoeExpertUpProjScale", ParameterKind.Input);

    public long LayerId { get; }

    public long HiddenSize { get; }

    public long IntermediateSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long IsNormTopkProb { get; }
}
