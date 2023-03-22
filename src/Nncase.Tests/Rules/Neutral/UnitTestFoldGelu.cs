// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Random;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

/// <inheritdoc />
[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldGelu : TransformTestBase
{
    public static TheoryData<int[]> FoldGeluWithScaleData => new()
    {
        new[] { 1, 3, 16, 16 },
        new[] { 1, 2, 4, 8 },
        new[] { 1, 1, 5, 5 },
    };

    public static TheoryData<int[]> FoldGeneralGeluData => new()
    {
        new[] { 1, 3, 16, 16 },
        new[] { 1, 2, 4, 8 },
        new[] { 1, 1, 5, 5 },
    };

    [Theory]
    [MemberData(nameof(FoldGeluWithScaleData))]
    public void TestFoldGeluWithScalePositive(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v0, 0.577350f); // "mul3Call"
            var v1 = IR.F.Math.Binary(BinaryOp.Div, v4, 1.414213f); // divCall
            var v2 = IR.F.NN.Erf(v1); // "erfCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Add, v2, 1f); // "addCall"
            var v5 = IR.F.Math.Binary(BinaryOp.Mul, v4, v3); // "mul2Call"
            var v6 = IR.F.Math.Binary(BinaryOp.Mul, v5, 0.5f); // "Mul1Call"
            rootPre = v6;
        }

        TestMatched<FoldGeluWithScale>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldGeluWithScaleData))]
    public void TestFoldGeluWithScaleNegative(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v0, 0.577350f); // "mul3Call"
            var v1 = IR.F.Math.Binary(BinaryOp.Div, v4, 1.414213f); // divCall
            var v2 = IR.F.NN.Erf(v1); // "erfCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Add, v2, 1f); // "addCall"
            var v5 = IR.F.Math.Binary(BinaryOp.Mul, v4, v3); // "mul2Call"
            var v6 = IR.F.Math.Binary(BinaryOp.Mod, v5, 0.5f); // "Mul1Call"
            rootPre = v6;
        }

        TestNotMatch<FoldGeluWithScale>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldGeneralGeluData))]
    public void TestFoldGeneralGeluPositive(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Div, v0, 1.4142135381698608f); // "divCall"
            var v2 = IR.F.NN.Erf(v1); // "erfCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Add, v2, 1f); // "addCall"
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v0, v3); // "mul2Call"
            var v5 = IR.F.Math.Binary(BinaryOp.Mul, v4, 0.5f); // "Mul1Call"
            rootPre = v5;
        }

        TestMatched<FoldGeneralGelu>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldGeneralGeluData))]
    public void TestFoldGeneralGeluNegative(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Div, v0, 1.4142135381698608f); // "divCall"
            var v2 = IR.F.NN.Erf(v1); // "erfCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Add, v2, 1f); // "addCall"
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v0, v3); // "mul2Call"
            var v5 = IR.F.Math.Binary(BinaryOp.Mod, v4, 0.5f); // "Mul1Call"
            rootPre = v5;
        }

        TestNotMatch<FoldGeneralGelu>(rootPre);
    }
}
