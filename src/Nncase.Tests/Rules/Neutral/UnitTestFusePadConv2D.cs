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

public class UnitTestFusePadConv2D
{
    private readonly RunPassOptions passOptions;

    public UnitTestFusePadConv2D()
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

    public static IEnumerable<object[]> TestFusePadConv2DPositiveData =>
        new[]
        {
            new object[] { new[] { 1, 1, 2, 2 }, new[,] { { 0, 0 },{ 0, 0 },{ 1, 1 },{ 1, 1 } }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 3, 1, 1, 1 } },
            new object[] { new[] { 1, 3, 4, 2 }, new[,] { { 0, 0 },{ 0, 0 },{ 5, 0 },{ 1, 3 } }, new[,] { { 0, 2 }, { 3, 2 } }, new[] { 1, 3, 2, 2 } },
        };

    [Theory]
    [MemberData(nameof(TestFusePadConv2DPositiveData))]
    public void TestFusePadConv2DPositive(int[] shape, int[,] pads1, int[,] pads2, int[] wShape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var w = Random.Normal(DataTypes.Float32, 0, 1, 0, wShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { wShape[0] });
        var rootPre = NN.Conv2D(NN.Pad(a, pads1, PadMode.Constant, 0), w, b, new[] { 1, 1 }, pads2, new[] { 1, 1 }, PadMode.Constant, 1);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldConstCall(),
            new FusePadConv2d()
        }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
