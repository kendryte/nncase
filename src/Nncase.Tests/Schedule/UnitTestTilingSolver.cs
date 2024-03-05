// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.Targets;
using Xunit;
using F = Nncase.IR.F;

namespace Nncase.Tests.ScheduleTest;

public class UnitTestTilingSolver
{
    [Fact]
    public void TestSimpleFor()
    {
        var input = Const.FromTensor(Tensor.FromScalar(1, new[] { 3, 16, 16 }));
        var rank = input.CheckedShape.Rank;

        var grid = IR.F.Affine.Grid(CPUTarget.Kind)
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(TIR.T.CreateBuffer(input.CheckedTensorType, TIR.MemoryLocation.Data, out _), AffineMap.Identity(rank), out _)
            .Body(IR.F.Math.Unary(UnaryOp.Abs, inTile))
            .Build();

        var call = CompilerServices.Tile(grid, new IRModule());
    }
}
