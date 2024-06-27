// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Reactive;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Xunit;
using F = Nncase.IR.F;

namespace Nncase.Tests.AffineTest;

public class UnitTestFor
{
    [Fact]
    public void TestAffineReverse()
    {
        // [m,l] @ [l,n] => [m,n],  [d0,d1,d2] -> [2d1 + 3, d2 - 1]
        var read = AffineRelation.FromCallable((dims, syms) => new AffineExpr[] { (2 * dims[1]) + 3, dims[2] - 1 }, 3, 0);

        System.Console.WriteLine(read);
        System.Console.WriteLine(read.Inverse());

        // [l,k] @ [k,n] => [l,n],  [l,k,n] -> [l,n]
        var write = AffineRelation.FromCallable((dims, syms) => new AffineExpr[] { dims[0], dims[2] }, 3, 0);
        System.Console.WriteLine(write.Inverse());
    }

    [Fact]
    public void TestSimpleFor()
    {
        int dimM = 32;
        int dimK = 64;
        int dimN = 32;

        var aAccessMap = AffineMap.Identity(2);
        var bAccessMap = aAccessMap;
        var cAccessMap = new AccessMap(
            new[]
            {
                AffineMap.FromCallable((AffineDomain m, AffineDomain n, AffineDomain k) => new AffineRange[] { new AffineRange(m.Offset, m.Extent), new AffineRange(k.Offset, k.Extent) }),
                AffineMap.FromCallable((AffineDomain m, AffineDomain n, AffineDomain k) => new AffineRange[] { new AffineRange(k.Offset, k.Extent), new AffineRange(n.Offset, n.Extent) }),
            },
            AffineMap.FromCallable((AffineDomain m, AffineDomain n) => new AffineRange[] { new AffineRange(m.Offset, m.Extent), new AffineRange(n.Offset, n.Extent), new AffineRange(F.Affine.Dim(2), F.Affine.Extent(2)) }));
        var a = Const.FromTensor(Tensor.FromScalar(1f, new[] { dimM, dimK }));
        var b = Const.FromTensor(Tensor.FromScalar(2f, new[] { dimK, dimN }));

        var aT2 = F.Affine.For(2, aAccessMap, a[aAccessMap]);
        var bT2 = F.Affine.For(2, bAccessMap, b[bAccessMap]);
        var cT2 = F.Affine.For(2, cAccessMap.Result, F.Tensors.MatMul(aT2[cAccessMap.Operands[0]], bT2[cAccessMap.Operands[1]]));
    }
}
