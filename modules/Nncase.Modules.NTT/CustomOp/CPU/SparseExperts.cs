// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.IR.CustomNTT;

[PatternFunctionalGenerator]
public sealed partial class SparseExperts : Op
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
    public static readonly ParameterInfo Extra = new(typeof(SparseExperts), 12, "Extra", ParameterKind.Input);

    public IRArray<int> QVectorizedAxes { get; }

    public IRArray<int> GateVectorizedAxes { get; }

    public IRArray<int> DownVectorizedAxes { get; }

    public IRArray<int> UpVectorizedAxes { get; }

    public IRArray<SBP> QSBPs { get; }

    public IRArray<SBP> GateSBPs { get; }

    public IRArray<SBP> DownSBPs { get; }

    public IRArray<SBP> UpSBPs { get; }

    public long HiddenSize { get; }

    public long MoEIntermediateSize { get; }

    public long NumExpert { get; }

    public long NumTopK { get; }

    public long ChunkSize { get; }

    public Cost Cost { get; }

    public string CSourcePath { get; }

    public string FuncName { get; }

    // public DataType OutputDataType { get; }
    public override string DisplayProperty() => $"QVectorizedAxes: {QVectorizedAxes}, GateVectorizedAxes: {GateVectorizedAxes}, DownVectorizedAxes: {DownVectorizedAxes}, UpVectorizedAxes: {UpVectorizedAxes}, HiddenSize: {HiddenSize}, MoEIntermediateSize: {MoEIntermediateSize}, NumExpert: {NumExpert}, NumTopK: {NumTopK}, ChunkSize: {ChunkSize}";
}
