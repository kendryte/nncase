// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Globalization;
using System.Text.RegularExpressions;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.Passes.Distributed;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
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
        var extra_size = /* node.ExtraWorkload; */ 10 * 1024 * 1024 / 2;

        int qAxis = 1;
        int wAxis = 2;

        // moeExpertGateProjW:          [num_expert, moe_intermediate_size, hidden_size]
        // moeExpertUpProjW:            [num_expert, moe_intermediate_size, hidden_size]
        // moeExpertDownProjW:          [num_expert, hidden_size, moe_intermediate_size]
        // 这里需要确保三个weights的Waxis维度在进行pack操作后是8的倍数，同时需要确保pack后的数据所对应的hiddensize或者moe_intermediate_size在三个weights中是相同的，因此需要先确定出哪个weights在pack后需要pad，然后去pad其他两个weights
        var moeExpertGateProjWShape = moeExpertGateProjW.CheckedShape.ToValueArray();
        var moeExpertDownProjWShape = moeExpertDownProjW.CheckedShape.ToValueArray();
        var moeExpertUpProjWShape = moeExpertUpProjW.CheckedShape.ToValueArray();

        static long Gcd(long a, long b)
        {
            while (b != 0)
            {
                (a, b) = (b, a % b);
            }

            return Math.Abs(a);
        }

        static long Lcm(long a, long b)
        {
            return a == 0 || b == 0 ? 0 : Math.Abs(a / Gcd(a, b) * b);
        }

        static Expr PadToLength(Expr tensor, params (int Axis, long TargetLength)[] axisTargets)
        {
            var shape = tensor.CheckedShape;
            var pads = IR.Shapes.Paddings.Zeros(shape.Rank).ToDimensionArray();
            var needPad = false;

            foreach (var (axis, targetLength) in axisTargets)
            {
                if (axis < 0 || axis >= shape.Rank)
                {
                    throw new ArgumentOutOfRangeException(nameof(axisTargets), axis, "Axis is out of range.");
                }

                var dim = shape[axis];
                if (!dim.IsFixed)
                {
                    throw new InvalidOperationException("Cannot pad tensor with dynamic dimension.");
                }

                var pad = targetLength - dim.FixedValue;
                if (pad < 0)
                {
                    throw new InvalidOperationException("Target length is smaller than current length.");
                }

                if (pad > 0)
                {
                    pads[axis, 1] = pad;
                    needPad = true;
                }
            }

            if (!needPad)
            {
                return tensor;
            }

            var padValue = Tensor.Zero(tensor.CheckedDataType);
            return IR.F.NN.Pad(tensor, Dimension.ConcatPadding(pads), PadMode.Constant, padValue);
        }

        var gateLanes = 128 / moeExpertGateProjW.CheckedDataType.SizeInBytes;
        var downLanes = 128 / moeExpertDownProjW.CheckedDataType.SizeInBytes;
        var upLanes = 128 / moeExpertUpProjW.CheckedDataType.SizeInBytes;

        var hiddenLanes = Lcm(gateLanes, upLanes);
        var hiddenAlignStep = hiddenLanes * 8;
        var intermediateAlignStep = downLanes * 8;

        var hiddenAligned = MathUtility.AlignUp(Math.Max(Math.Max(moeExpertGateProjWShape[wAxis], moeExpertUpProjWShape[wAxis]), moeExpertDownProjWShape[1]), hiddenAlignStep);
        var intermediateAligned = MathUtility.AlignUp(Math.Max(Math.Max(moeExpertDownProjWShape[wAxis], moeExpertGateProjWShape[1]), moeExpertUpProjWShape[1]), intermediateAlignStep);

        var paddedGateProjW = PadToLength(moeExpertGateProjW, (1, intermediateAligned), (wAxis, hiddenAligned));
        var paddedDownProjW = PadToLength(moeExpertDownProjW, (1, hiddenAligned), (wAxis, intermediateAligned));
        var paddedUpProjW = PadToLength(moeExpertUpProjW, (1, intermediateAligned), (wAxis, hiddenAligned));

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
            paddedGateProjW.CheckedShape[1].FixedValue,
            sparseExperts.NumExpert,
            sparseExperts.NumTopK,
            sparseExperts.ChunkSize,
            new() { [CostFactorNames.CPUCycles] = node.Cost },
            node.CSourcePath,
            node.FuncName);

        var updatedCall = call.With(
            target: target,
            arguments: new[]
            {
                IR.F.Tensors.Transpose(IR.F.Tensors.Pack(q, new[] { 128 / q.CheckedDataType.SizeInBytes }, new[] { qAxis }), new[] { 1, 0 }),
                routerIdx,
                routerWeights,
                moeExpertGateInputScale,
                IR.F.Tensors.Transpose(
                    IR.F.Tensors.Pack(paddedGateProjW, new[] { 128 / moeExpertGateProjW.CheckedDataType.SizeInBytes }, new[] { wAxis }),
                    new[] { 0, 2, 1 }),
                moeExpertGateProjScale,
                moeExpertDownInputScale,
                IR.F.Tensors.Transpose(
                    IR.F.Tensors.Pack(paddedDownProjW, new[] { 128 / moeExpertDownProjW.CheckedDataType.SizeInBytes }, new[] { wAxis }),
                    new[] { 0, 2, 1 }),
                moeExpertDownProjScale,
                moeExpertUpInputScale,
                IR.F.Tensors.Transpose(
                    IR.F.Tensors.Pack(paddedUpProjW, new[] { 128 / moeExpertUpProjW.CheckedDataType.SizeInBytes }, new[] { wAxis }),
                    new[] { 0, 2, 1 }),
                moeExpertUpProjScale,
                IR.F.Buffer.Uninitialized(DataTypes.UInt8, TIR.MemoryLocation.Data, [extra_size]),
            },
            metadata: call.Metadata);
        return IR.F.Tensors.Cast(IR.F.Tensors.Unpack(IR.F.Tensors.Transpose(updatedCall, new[] { 1, 0 }), new[] { 128 / q.CheckedDataType.SizeInBytes }, new[] { qAxis }), q.CheckedDataType);
    }
}
