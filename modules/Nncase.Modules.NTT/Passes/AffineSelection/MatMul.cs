// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectMatMul(Op op, Call call, Expr output)
    {
        var lhs = (Expr)call.Arguments[IR.Math.MatMul.Lhs.Index];
        var rhs = (Expr)call.Arguments[IR.Math.MatMul.Rhs.Index];
        bool reduceSum = false;

        // TODO: summa not support tiling for now.
        if (lhs.CheckedType is DistributedType dta &&
            rhs.CheckedType is DistributedType dtb)
        {
            if (op is IR.NTT.VectorizedMatMul pmm)
            {
                var dinfo = pmm.GetDimInfo(dta.TensorType.Shape.Rank, dtb.TensorType.Shape.Rank);
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dta.AxisPolicies[dinfo.Lk] == dtb.AxisPolicies[dinfo.Rn] &&
                    dta.AxisPolicies[dinfo.Lm] == dtb.AxisPolicies[dinfo.Rk])
                {
                    return call;
                }

                if (dta.AxisPolicies[dinfo.Lk] == dtb.AxisPolicies[dinfo.Rk] && dta.AxisPolicies[dinfo.Lk] is SBPSplit)
                {
                    reduceSum = true;
                }
            }
            else if (op is IR.Math.MatMul)
            {
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dta.AxisPolicies[^2] == dtb.AxisPolicies[^2] &&
                    dta.AxisPolicies[^1] == dtb.AxisPolicies[^1])
                {
                    return call;
                }

                if (dta.AxisPolicies[^1] == dtb.AxisPolicies[^2] && dta.AxisPolicies[^1] is SBPSplit)
                {
                    reduceSum = true;
                }
            }
        }

        if (reduceSum)
        {
            return call;
        }

        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var rank = Math.Max(lhsShape.Rank, rhsShape.Rank) + 1;
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhsShape.Rank];
        var rhsRes = new AffineRange[rhsShape.Rank];
        for (int i = rank - 1 - 3; i >= 0; i--)
        {
            var lhsi = i - (rank - (lhsShape.Rank + 1));
            var rhsi = i - (rank - (rhsShape.Rank + 1));
#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly
            switch (lhsi, rhsi)
            {
                case ( >= 0, >= 0):
                    switch (lhsShape[lhsi], rhsShape[rhsi])
                    {
                        case (DimConst cDimA, DimConst cDimB):
                            switch (cDimA.Value, cDimB.Value)
                            {
                                case (long a, long b) when a == b:
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                case (1, _):
                                    lhsRes[lhsi] = new AffineRange(0, 1);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                case (_, 1):
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(0, 1);
                                    break;
                                default:
                                    return call;
                            }

                            break;

                        case (DimConst constA, Dimension dimB):
                            switch (constA.Value, dimB.Metadata.Range)
                            {
                                case (1, { Min: >= 1 }):
                                    lhsRes[lhsi] = new AffineRange(0, 1);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        case (Dimension dimA, DimConst constB):
                            switch (dimA.Metadata.Range, constB.Value)
                            {
                                case ({ Min: >= 1 }, 1):
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(0, 1);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        case (Dimension dimA, Dimension dimB):
                            switch (dimA.Metadata.Range, dimB.Metadata.Range)
                            {
                                case (var rangeA, var rangeB) when rangeA == rangeB:
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        default:
                            return call;
                    }

                    break;
                case ( < 0, >= 0):
                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case ( >= 0, < 0):
                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case ( < 0, < 0):
                    break;
            }
#pragma warning restore SA1008 // Opening parenthesis should be spaced correctly
        }

        var (om, ok, on) = (rank - 3, rank - 2, rank - 1);
        var (lm, lk) = (lhsShape.Rank - 2, lhsShape.Rank - 1);
        var (rk, rn) = (rhsShape.Rank - 2, rhsShape.Rank - 1);
        if (op is IR.NTT.VectorizedMatMul pm)
        {
            if (pm.TransposeA)
            {
                (lm, lk) = (lk, lm);
            }

            if (pm.TransposeB)
            {
                (rk, rn) = (rn, rk);
            }
        }
        else if (op is IR.NTT.PackedMatMul)
        {
            // Transpose B
            (rk, rn) = (rn, rk);
        }

        lhsRes[lm] = new AffineRange(domains[om].Offset, domains[om].Extent);
        lhsRes[lk] = new AffineRange(domains[ok].Offset, domains[ok].Extent);
        rhsRes[rk] = lhsRes[lk];
        rhsRes[rn] = new AffineRange(domains[on].Offset, domains[on].Extent);

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outMap = new AffineMap(domains, default, domains.SkipLast(3).Concat([domains[om], domains[on]]).Select(x => new AffineRange(x.Offset, x.Extent)).ToArray());
        return IR.F.Affine.Grid()
            .Domain(rank, out var domainVar)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(output, outMap, out var outTile)
            .Body(op switch
            {
                IR.Math.MatMul => TIR.F.NTT.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[ok][0], 0L)),
                IR.NTT.VectorizedMatMul pop => TIR.F.NTT.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[ok][0], 0L), pop.LhsVectorizedAxes, pop.RhsVectorizedAxes, pop.TransposeA, pop.TransposeB, pop.FusedReduce),
                IR.NTT.PackedMatMul pop => TIR.F.NTT.PackedMatMul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[ok][0], 0L), pop.FusedReduce),
                _ => throw new System.Diagnostics.UnreachableException(),
            }).Build();
    }
}
