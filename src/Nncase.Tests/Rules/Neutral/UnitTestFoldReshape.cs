using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using Nncase.IR.F;
using Random = Nncase.IR.F.Random;
using Math = Nncase.IR.F.Math;

namespace Nncase.Tests.Rules.Neutral;

public class UnitTestFoldReshape
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldReshape()
    {
        string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "..", "tests_output");
        dumpDir = Path.GetFullPath(dumpDir);
        Directory.CreateDirectory(dumpDir);
        passOptions = new RunPassOptions(null, 3, dumpDir);
    }

    private static string GetThisFilePath([CallerFilePath] string path = null)
    {
        return path;
    }

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
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(a, newShape);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopReshape() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
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
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Reshape(Tensors.Reshape(a, newShape1), newShape2);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoReshapes() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
