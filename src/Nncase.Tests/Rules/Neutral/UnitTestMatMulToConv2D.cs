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

public class UnitTestMatMulToConv2D
{
    private readonly RunPassOptions passOptions;

    public UnitTestMatMulToConv2D()
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

    public static IEnumerable<object[]> TestMatMulToConv2DPositiveData =>
        new[]
        {
            new object[] { new[] { 5, 4 }, new[] { 4, 6 } },
            new object[] { new[] { 1, 7 }, new[] { 7, 12 } },
        };

    [Theory]
    [MemberData(nameof(TestMatMulToConv2DPositiveData))]
    public void TestFusePadConv2DPositive(int[] aShape, int[] bShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, bShape);
        var rootPre = Math.MatMul(a, b);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldConstCall(),
            new FusePadConv2d()
        }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
