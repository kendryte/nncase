// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public static bool TryGetCastAffineRelation(DataType inDType, DataType outDType, IRArray<int> vectorizedAxes, IR.Shape inShape, IR.Shape outShape, out AffineDomain[] domains, out AffineMap inMap, out AffineMap outMap)
    {
        domains = Array.Empty<AffineDomain>();
        inMap = null!;
        outMap = null!;

        if (inShape is not { Rank: > 0 })
        {
            return false;
        }

        var rank = inShape.Rank;
        var (inScale, outScale) = (inDType, outDType) switch
        {
            (PrimType, PrimType) => (1, 1),
            (VectorType { ElemType: var a }, VectorType { ElemType: var b }) when a.SizeInBytes >= b.SizeInBytes => (a.SizeInBytes / b.SizeInBytes, 1),
            (VectorType { ElemType: var a }, VectorType { ElemType: var b }) when a.SizeInBytes < b.SizeInBytes => (1, b.SizeInBytes / a.SizeInBytes),
            _ => throw new NotSupportedException($"Unsupported cast from {inDType} to {outDType}."),
        };

        domains = IR.F.Affine.Domains(rank);
        inMap = inMap = new AffineMap(domains, default, domains.Select((x, i) =>
        {
            if (vectorizedAxes.Contains(i))
            {
                return new AffineRange(x.Offset * inScale, x.Extent * inScale);
            }
            else
            {
                return new AffineRange(x.Offset, x.Extent);
            }
        }).ToArray());
        outMap = outMap = new AffineMap(domains, default, domains.Select((x, i) =>
        {
            if (vectorizedAxes.Contains(i))
            {
                return new AffineRange(x.Offset * outScale, x.Extent * outScale);
            }
            else
            {
                return new AffineRange(x.Offset, x.Extent);
            }
        }).ToArray());
        return true;
    }

    public Expr SelectCast(IR.Tensors.Cast cast, Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Cast.Input];
        var (inDType, outDType) = (input.CheckedDataType, cast.NewType);
        var vectorizedAxes = Array.Empty<int>();
        if (!TryGetCastAffineRelation(inDType, outDType, vectorizedAxes, input.CheckedShape, output.CheckedShape, out var domains, out var inMap, out var outMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inMap, out var inTile)
            .Write(output, outMap, out var outTile)
            .Body(TIR.F.NTT.Cast(inTile, outTile, cast.NewType, cast.CastMode, Array.Empty<int>(), None.Default))
            .Build();
    }

    public Expr SelectVectorizedCast(IR.NTT.VectorizedCast cast, Call call, Expr output)
    {
        var input = (Expr)call[IR.NTT.VectorizedCast.Input];
        var (inDType, outDType) = (input.CheckedDataType, cast.NewType);
        var vectorizedAxes = cast.VectorizeAxes;
        if (!TryGetCastAffineRelation(inDType, outDType, vectorizedAxes, input.CheckedShape, output.CheckedShape, out var domains, out var inMap, out var outMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(input, inMap, out var inTile)
            .Write(output, outMap, out var outTile)
            .Body(TIR.F.NTT.Cast(inTile, outTile, cast.NewType, cast.CastMode, vectorizedAxes, (Expr)call[IR.NTT.VectorizedCast.PostOps]))
            .Build();
    }
}
