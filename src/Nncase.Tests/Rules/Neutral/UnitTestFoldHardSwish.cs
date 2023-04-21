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
public class UnitTestFoldHardSwish : TransformTestBase
{
    public static TheoryData<int[]> FoldGeneralHardSwishData => new()
    {
        new[] { 1, 3, 16, 16 },
        new[] { 1, 2, 4, 8 },
        new[] { 1, 1, 5, 5 },
    };

    [Theory]
    [MemberData(nameof(FoldGeneralHardSwishData))]
    public void TestFoldHardSwish1Positive(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Add, v0, 3f); // "addCall"
            var v2 = IR.F.Math.Clamp(v1, new ValueRange<float>(0f, 6f)); // "clampCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Mul, v0, v2); // "mulCall"
            var v4 = IR.F.Math.Binary(BinaryOp.Div, v3, 6f); // "divCall"
            rootPre = v4;
        }

        TestMatched<FoldHardSwish1>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldGeneralHardSwishData))]
    public void TestFoldHardSwish1Negative(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Add, v0, 3f); // "addCall"
            var v2 = IR.F.Math.Clamp(v1, new ValueRange<float>(0f, 6f)); // "clampCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Mul, v2, v0); // "mulCall"
            var v4 = IR.F.Math.Binary(BinaryOp.Div, v3, 6f); // "divCall"
            rootPre = v4;
        }

        TestNotMatch<FoldHardSwish1>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldGeneralHardSwishData))]
    public void TestFoldHardSwish2Positive(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Add, v0, 3f); // "addCall"
            var v2 = IR.F.NN.Relu6(v1); // "relu6Call"
            var v3 = IR.F.Math.Binary(BinaryOp.Mul, 0.1666666716337204f, v2); // "mul1_6Call"
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v3, v0); // "mulCall"
            rootPre = v4;
        }

        TestMatched<FoldHardSwish2>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldGeneralHardSwishData))]
    public void TestFoldHardSwish2Negative(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Math.Binary(BinaryOp.Add, v0, 3f); // "addCall"
            var v2 = IR.F.Math.Clamp(v1, new ValueRange<float>(0f, 6f)); // "clampCall"
            var v3 = IR.F.Math.Binary(BinaryOp.Mul, v2, v0); // "mulCall"
            var v4 = IR.F.Math.Binary(BinaryOp.Div, v3, 6f); // "divCall"
            rootPre = v4;
        }

        TestNotMatch<FoldHardSwish2>(rootPre);
    }
}
