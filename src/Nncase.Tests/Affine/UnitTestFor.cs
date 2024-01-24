// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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
                AffineMap.FromCallable((AffineDim m, AffineDim n, AffineDim k) => new[] { m, k }),
                AffineMap.FromCallable((AffineDim m, AffineDim n, AffineDim k) => new[] { k, n }),
            },
            AffineMap.FromCallable((AffineDim m, AffineDim n) => new[] { m, n, F.Affine.Dim(2) }));
        var a = Const.FromTensor(Tensor.FromScalar(1f, new[] { dimM, dimK }));
        var b = Const.FromTensor(Tensor.FromScalar(2f, new[] { dimK, dimN }));

        var aT2 = F.Affine.For(2, aAccessMap, a[aAccessMap]);
        var bT2 = F.Affine.For(2, bAccessMap, b[bAccessMap]);
        var cT2 = F.Affine.For(2, cAccessMap.Result, F.Tensors.MatMul(aT2[cAccessMap.Operands[0]], bT2[cAccessMap.Operands[1]]));
    }
}
