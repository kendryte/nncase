// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Globalization;
using System.Text.RegularExpressions;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes.Distributed;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT.CustomOp;

[RuleGenerator]
public partial class ToCustomSparseExperts : RewriteRule<Pattern>
{
    public ToCustomSparseExperts(CustomOpScheme scheme)
    {
        Scheme = scheme;
    }

    public ToCustomSparseExperts()
    {
        Scheme = null!;
    }

    public CustomOpScheme Scheme { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.NN.IsSparseExperts(
        "sparseExperts",
        "call",
        _ => true,
        IsWildcard("q"),
        IsWildcard("routerIdx"),
        IsWildcard("routerWeights"),
        IsWildcard("moeExpertGateInputScale"),
        IsWildcard("moeExpertGateProjW"),
        IsWildcard("moeExpertGateProjScale"),
        IsWildcard("moeExpertDownInputScale"),
        IsWildcard("moeExpertDownProjW"),
        IsWildcard("moeExpertDownProjScale"),
        IsWildcard("moeExpertUpInputScale"),
        IsWildcard("moeExpertUpProjW"),
        IsWildcard("moeExpertUpProjScale"));

    private Expr? GetReplace(
        Call call,
        SparseExperts sparseExperts,
        Expr q,
        Expr routerIdx,
        Expr routerWeights,
        Expr moeExpertGateInputScale,
        Expr moeExpertGateProjW,
        Expr moeExpertGateProjScale,
        Expr moeExpertDownInputScale,
        Expr moeExpertDownProjW,
        Expr moeExpertDownProjScale,
        Expr moeExpertUpInputScale,
        Expr moeExpertUpProjW,
        Expr moeExpertUpProjScale)
    {
        if (Scheme is null)
        {
            return null;
        }

        // Name pattern
        var node = Scheme.Outputs.FirstOrDefault(op => call.Metadata.OutputNames?[0] is string outputName && Regex.IsMatch(outputName, op.Name ?? string.Empty));

#if false
        if (node is null)
        {
            node = Scheme.Outputs.FirstOrDefault(op =>
                op.Op.ToLower(CultureInfo.CurrentCulture) == "SparseExperts" &&
                op.Shape[0].SequenceEqual(lhs.CheckedShape.ToValueArray()) &&
                op.Shape[1].SequenceEqual(rhs.CheckedShape.ToValueArray()));
        }
#endif

        if (node is null)
        {
            return null;
        }

        var qSbp = node.SBP.Length > 0 ? node.SBP[0] : Array.Empty<IR.SBP>();
        var gateSbp = node.SBP.Length > 1 ? node.SBP[1] : Array.Empty<IR.SBP>();
        var downSbp = node.SBP.Length > 2 ? node.SBP[2] : Array.Empty<IR.SBP>();
        var upSbp = node.SBP.Length > 3 ? node.SBP[3] : Array.Empty<IR.SBP>();
        var extra_size = /* node.ExtraWorkload; */ 1000;

        int qAxis = 1;
        int wAxis = 2;
        var target = new IR.CustomNTT.SparseExperts(
            new[] { qAxis },
            new[] { wAxis },
            new[] { wAxis },
            new[] { wAxis },
            qSbp,
            gateSbp,
            downSbp,
            upSbp,
            sparseExperts.HiddenSize,
            sparseExperts.MoEIntermediateSize,
            sparseExperts.NumExpert,
            sparseExperts.NumTopK,
            sparseExperts.ChunkSize,
            new() { [CostFactorNames.CPUCycles] = node.Cost },
            node.CSourcePath,
            node.FuncName);

        return call.With(
            target: target,
            arguments: new[]
            {
                IR.F.Tensors.Transpose(IR.F.Tensors.Pack(q, new[] { 128 / q.CheckedDataType.SizeInBytes }, new[] { qAxis }), new[] { 1, 0 }),
                routerIdx,
                routerWeights,
                moeExpertGateInputScale,
                IR.F.Tensors.Transpose(
                    IR.F.Tensors.Pack(moeExpertGateProjW, new[] { 128 / moeExpertGateProjW.CheckedDataType.SizeInBytes }, new[] { wAxis }),
                    new[] { 0, 2, 1 }),
                moeExpertGateProjScale,
                moeExpertDownInputScale,
                IR.F.Tensors.Transpose(
                    IR.F.Tensors.Pack(moeExpertDownProjW, new[] { 128 / moeExpertDownProjW.CheckedDataType.SizeInBytes }, new[] { wAxis }),
                    new[] { 0, 2, 1 }),
                moeExpertDownProjScale,
                moeExpertUpInputScale,
                IR.F.Tensors.Transpose(
                    IR.F.Tensors.Pack(moeExpertUpProjW, new[] { 128 / moeExpertUpProjW.CheckedDataType.SizeInBytes }, new[] { wAxis }),
                    new[] { 0, 2, 1 }),
                moeExpertUpProjScale,
                IR.F.Buffer.Uninitialized(DataTypes.UInt8, TIR.MemoryLocation.Data, [extra_size]),
            },
            metadata: call.Metadata);
    }
}
