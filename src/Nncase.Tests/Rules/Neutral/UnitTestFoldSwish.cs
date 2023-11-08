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
using Nncase.IR.Math;
using Nncase.IR.Random;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Random = Nncase.IR.F.Random;
using Sigmoid = Nncase.IR.NN.Sigmoid;

namespace Nncase.Tests.Rules.NeutralTest;

/// <inheritdoc />
[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldSwish : TransformTestBase
{
    public static TheoryData<int[]> FoldSwishData => new()
    {
        new[] { 1, 3, 16, 16 },
        new[] { 1, 2, 4, 8 },
        new[] { 1, 1, 5, 5 },
    };

    [Theory]
    [MemberData(nameof(FoldSwishData))]
    public void TestFoldSwishPattern1Positive1(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.NN.Sigmoid(v0);
            var v2 = IR.F.Math.Binary(BinaryOp.Mul, v1, v0);
            rootPre = v2;
        }

        TestMatched<FoldSwishPattern1>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldSwishData))]
    public void TestFoldSwishPattern1Negative1(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.NN.Sigmoid(v0);
            var v2 = IR.F.Math.Binary(BinaryOp.Add, v1, v0);
            rootPre = v2;
        }

        TestNotMatch<FoldSwishPattern1>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldSwishData))]
    public void TestFoldSwishPattern2Positive2(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v0_2 = IR.F.Math.Binary(BinaryOp.Mul, v0, 2.0f);
            var v1 = IR.F.NN.Sigmoid(v0_2);
            var v2 = IR.F.Math.Binary(BinaryOp.Mul, v0, v1);
            rootPre = v2;
        }

        TestMatched<FoldSwishPattern2>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldSwishData))]
    public void TestFoldSwishPattern2Negative2(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.NN.Sigmoid(v0);
            var v2 = IR.F.Math.Binary(BinaryOp.Add, v0, v1);
            rootPre = v2;
        }

        TestNotMatch<FoldSwishPattern2>(rootPre);
    }
}
