// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

public sealed partial class SparseExperts : Nncase.TIR.NTT.NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(SparseExperts), 0, "q", ParameterKind.Input);
    public static readonly ParameterInfo RouterIdx = new(typeof(SparseExperts), 1, "RouterIdx", ParameterKind.Input);
    public static readonly ParameterInfo RouterWeights = new(typeof(SparseExperts), 2, "RouterWeights", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertGateInputScale = new(typeof(SparseExperts), 3, "MoeExpertGateInputScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertGateProjW = new(typeof(SparseExperts), 4, "MoeExpertGateProjW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertGateProjScale = new(typeof(SparseExperts), 5, "MoeExpertGateProjScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertDownInputScale = new(typeof(SparseExperts), 6, "MoeExpertDownInputScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertDownProjW = new(typeof(SparseExperts), 7, "MoeExpertDownProjW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertDownProjScale = new(typeof(SparseExperts), 8, "MoeExpertDownProjScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertUpInputScale = new(typeof(SparseExperts), 9, "MoeExpertUpInputScale", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertUpProjW = new(typeof(SparseExperts), 10, "MoeExpertUpProjW", ParameterKind.Input);
    public static readonly ParameterInfo MoeExpertUpProjScale = new(typeof(SparseExperts), 11, "MoeExpertUpProjScale", ParameterKind.Input);

    public long HiddenSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long ChunkSize { get; }
}
