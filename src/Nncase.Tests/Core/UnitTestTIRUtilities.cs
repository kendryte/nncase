// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;
using OrtKISharp;
using Xunit;
using Range = Nncase.TIR.Range;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTIRUtilities
{
    [Fact]
    public void TestComputePaddings()
    {
        // Arrange
        IReadOnlyList<TIR.Range> bounds = new List<TIR.Range>()
        {
            new TIR.Range(0, 32, 1), new TIR.Range(0, 16, 1),
            new TIR.Range(0, 8, 1),
        };

        IReadOnlyList<TIR.Range> targetbounds = new List<TIR.Range>()
        {
            new TIR.Range(0, 32, 1), new TIR.Range(0, 8, 1),
            new TIR.Range(0, 8, 1),
        };

        var shape = new IR.Shape(new List<int>() { 32, 16, 8 });

        // Act
        var paddings1 = TIRUtilities.ComputePaddings(bounds, shape);
        var paddings2 = TIRUtilities.ComputePaddings(bounds, targetbounds);

        // Assert
        Assert.Equal(3, paddings1.Count);
        Assert.Equal(3, paddings2.Count);
    }

    [Fact]
    public void TestComputeNoPadBounds()
    {
        // Arrange
        IReadOnlyList<TIR.Range> bounds = new List<TIR.Range>()
        {
            new TIR.Range(0, 32, 1), new TIR.Range(0, 16, 1),
            new TIR.Range(0, 8, 1),
        };
        IReadOnlyList<TIR.Range> sub_bounds = new List<TIR.Range>()
        {
            new TIR.Range(0, 32, 1), new TIR.Range(0, 16, 1),
            new TIR.Range(0, 16, 1),
        };
        IReadOnlyList<(IR.Expr Before, IR.Expr After)> paddings = new List<(IR.Expr Before, IR.Expr After)>()
        {
            (IR.F.Math.Max(-1, 0), IR.F.Math.Max(0 - 32 - 1, 0)),
            (IR.F.Math.Max(0, 0), IR.F.Math.Max(0 - 16 - 0, 0)),
            (IR.F.Math.Max(-2, 0), IR.F.Math.Max(0 - 8 - 2, 0)),
        };
        IReadOnlyList<(IR.Expr Before, IR.Expr After)> sub_paddings = new List<(IR.Expr Before, IR.Expr After)>()
        {
            (IR.F.Math.Max(-1, 0), IR.F.Math.Max(0 - 32 - 1, 0)),
            (IR.F.Math.Max(0, 0), IR.F.Math.Max(0 - 16 - 0, 0)),
            (IR.F.Math.Max(-2, 0), IR.F.Math.Max(0 - 8 - 2, 0)),
        };

        // Act
        var noPadBounds = TIRUtilities.ComputeNoPadBounds(bounds, paddings);

        // Assert
        Assert.Equal(3, noPadBounds.Count);
    }

    [Fact]
    public void TestClampBounds()
    {
        // Arrange
        IReadOnlyList<TIR.Range> bounds = new List<TIR.Range>()
        {
            new TIR.Range(-3, 36, 1), new TIR.Range(0, 20, 1),
            new TIR.Range(0, 10, 1),
        };
        var shape = new IR.Shape(new List<int>() { 32, 16, 8 });

        // Act
        var clampedBounds1 = TIRUtilities.ClampBounds(bounds, shape);

        // Assert
        Assert.Equal(3, clampedBounds1.Count);
    }

    [Fact]
    public void TestComputeBounds()
    {
        // Arrange
        IReadOnlyList<TIR.Range> sub_no_pad_bounds = new List<TIR.Range>()
        {
            new TIR.Range(0, 32, 1), new TIR.Range(0, 16, 1),
            new TIR.Range(0, 8, 1),
        };

        IReadOnlyList<TIR.Range> bounds = new List<TIR.Range>()
        {
            new TIR.Range(0, 32, 1), new TIR.Range(0, 8, 1),
            new TIR.Range(0, 8, 1),
        };
        IReadOnlyList<(IR.Expr Before, IR.Expr After)> paddings = new List<(IR.Expr Before, IR.Expr After)>()
        {
            (IR.F.Math.Max(-1, 0), IR.F.Math.Max(0 - 32 - 1, 0)),
            (IR.F.Math.Max(0, 0), IR.F.Math.Max(0 - 16 - 0, 0)),
            (IR.F.Math.Max(-2, 0), IR.F.Math.Max(0 - 8 - 2, 0)),
        };

        // Act
        var computeBounds = TIRUtilities.ComputeBounds(sub_no_pad_bounds, bounds, paddings);

        // Assert
        Assert.Equal(3, computeBounds.Count);
    }
}
