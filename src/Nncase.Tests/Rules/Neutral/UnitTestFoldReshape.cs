﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.F;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestFoldReshape : TestFixture.UnitTestFixtrue
{

    public static IEnumerable<object[]> TestFoldNopReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 4 }, new[] { 4 } },
            new object[] { new[] { 2, 3 }, new[] { 2, 3 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopReshapePositiveData))]
    public void TestFoldNopReshapePositive(int[] shape, int[] newShape)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopReshape() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldNopReshapeNegativeData =>
        new[]
        {
            new object[] { new[] { 4 }, new[] { 2, 2 } },
            new object[] { new[] { 2, 3 }, new[] { 3, 2 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopReshapeNegativeData))]
    public void TestFoldNopReshapeNegative(int[] shape, int[] newShape)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopReshape() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldTwoReshapesPositiveData =>
        new[]
        {
            new object[] { new[] { 4 }, new[] { 2, 2 }, new[] { 1, 4 } },
            new object[] { new[] { 2, 4 }, new[] { 8 }, new[] { 4, 2 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoReshapesPositiveData))]
    public void TestFoldTwoReshapesPositive(int[] shape, int[] newShape1, int[] newShape2)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(Tensors.Reshape(a, newShape1), newShape2);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoReshapes() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
