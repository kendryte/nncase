using System;
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

public class UnitTestFoldTranspose : TestFixture.UnitTestFixtrue
{

    public static IEnumerable<object[]> TestFoldNopTransposePositiveData =>
        new[]
        {
            new object[] { new[] { 2, 4 }, new[] { 0, 1 } },
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0, 1, 2, 3 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopTransposePositiveData))]
    public void TestFoldNopTransposePositive(int[] shape, int[] perm)
    {
        var caseOptions = passOptions.IndentDir($"FoldNop_{string.Join("_", shape)}_{string.Join("_", perm)}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(a, perm);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopTranspose() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldTwoTransposesPositiveData =>
        new[]
        {
            new object[] { new[] { 2, 4 }, new[] { 1, 0 }, new[] { 0, 1 } },
            new object[] { new[] { 2, 4, 6 }, new[] { 0, 2, 1 }, new[] {  1, 2, 0 } },
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0, 2, 3, 1 }, new[] { 3, 1, 2, 0 } },
            new object[] { new[] { 2, 4, 6, 8, 2}, new[] { 0, 2, 3, 1, 4 }, new[] { 3, 1, 2, 4, 0 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoTransposesPositiveData))]
    public void TestFoldTwoTransposesPositive(int[] shape, int[] perm1, int[] perm2)
    {
        var caseOptions = passOptions.IndentDir($"FoldTwo_{string.Join("_", shape)}_{string.Join("_", perm1)}_{string.Join("_", perm2)}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(Tensors.Transpose(a, perm1), perm2);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoTransposes() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestTransposeToReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 1, 2, 4 }, new[] { 1, 0, 2 } },
            new object[] { new[] { 2, 1, 6, 1 }, new[] { 1, 0, 3, 2 } },
        };

    [Theory]
    [MemberData(nameof(TestTransposeToReshapePositiveData))]
    public void TestTransposeToReshapePositive(int[] shape, int[] perm)
    {
        var caseOptions = passOptions.IndentDir($"FoldTToR_{string.Join("_", shape)}_{string.Join("_", perm)}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Transpose(a, perm);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldShapeOf(),
            new TransposeToReshape()
        }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
