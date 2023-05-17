// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR.F;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldTranspose : TransformTestBase
{
    public static IEnumerable<object[]> TestFoldNopTransposePositiveData =>
        new[]
        {
            new object[] { new[] { 2, 4 }, new[] { 0, 1 } },
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0, 1, 2, 3 } },
        };

    public static TheoryData<int, int[], int[], int[]> TestFoldTwoTransposesPositiveData => new TheoryData<int, int[], int[], int[]>
    {
        { 0, new[] { 2, 4 }, new[] { 1, 0 }, new[] { 0, 1 } },
        { 1, new[] { 2, 4, 6 }, new[] { 0, 2, 1 }, new[] { 1, 2, 0 } },
        { 2, new[] { 2, 4, 6, 8 }, new[] { 0, 2, 3, 1 }, new[] { 3, 1, 2, 0 } },
        { 3, new[] { 2, 4, 6, 8, 2 }, new[] { 0, 2, 3, 1, 4 }, new[] { 3, 1, 2, 4, 0 } },
        { 4, new[] { 1, 32, 112, 112 }, new[] { 0, 2, 3, 1 }, new[] { 0, 3, 1, 2 } },
    };

    public static IEnumerable<object[]> TestTransposeToReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 1, 2, 4 }, new[] { 1, 0, 2 } },
            new object[] { new[] { 2, 1, 6, 1 }, new[] { 1, 0, 3, 2 } },
        };

    public static TheoryData<(int Count, IR.Expr Act, int[] Perm)> TestCombineTransposeActivationsPositiveData => new()
    {
        (1, IR.F.NN.Relu(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 2, 4 })), new int[] { 1, 0, 2 }),
        (2, IR.F.NN.Celu(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), 0.6f), new int[] { 1, 0, 2 }),
        (3, IR.F.NN.HardSigmoid(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), 0.6f, 0.3f), new int[] { 1, 0, 2 }),
        (3, IR.F.NN.Erf(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 })), new int[] { 1, 0, 2 }),
        (3, IR.F.NN.Gelu(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), 1f), new int[] { 1, 0, 2 }),
    };

    public static TheoryData<(int Count, IR.Expr Act, int[] Perm)> TestCombineTransposeActivationsNegativeData => new()
    {
        (1, IR.F.NN.Softplus(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 2, 4 })), new int[] { 1, 0, 2 }),
        (2, IR.F.NN.Softsign(IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 })), new int[] { 1, 0, 2 }),
    };

    [Theory]
    [MemberData(nameof(TestFoldNopTransposePositiveData))]
    public void TestFoldNopTransposePositive(int[] shape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(a, perm);
        TestMatched<FoldNopTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldTwoTransposesPositiveData))]
    public void TestFoldTwoTransposesPositive(int count, int[] shape, int[] perm1, int[] perm2)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(Tensors.Transpose(a, perm1), perm2);
        TestMatched<FoldTwoTransposes>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestTransposeToReshapePositiveData))]
    public void TestTransposeToReshapePositive(int[] shape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(a, perm);
        TestMatchedCore(
            rootPre,
            rules: new IRewriteRule[]
            {
                new FoldShapeOf(),
                new TransposeToReshape(),
            });
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeActivationsPositiveData))]
    public void TestCombineTransposeActivationsPositive((int Count, IR.Expr Act, int[] Perm) param)
    {
        using var dumpScope = new DumpScope($"{param.Count}");
        var rootPre = Tensors.Transpose(param.Act, param.Perm);
        TestMatched<CombineTransposeActivations>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeActivationsNegativeData))]
    public void TestCombineTransposeActivationsNegative((int Count, IR.Expr Act, int[] Perm) param)
    {
        using var dumpScope = new DumpScope($"{param.Count}");
        var rootPre = Tensors.Transpose(param.Act, param.Perm);
        TestNotMatch<CombineTransposeActivations>(rootPre);
    }
}
