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

public class UnitTestFoldPad
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldPad()
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

    public static IEnumerable<object[]> TestFoldNopPadPositiveData =>
        new[]
        {
            new object[] { new[] { 1 }, (Tensor)new[] { 0, 0 } },
            new object[] { new[] { 1, 1 }, (Tensor)new[] { 0, 0, 0, 0 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopPadPositiveData))]
    public void TestFoldNopPadPositive(int[] shape, Tensor pads)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(a, pads, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopPad() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldTwoPadsPositiveData =>
        new[]
        {
            new object[] { new[] { 1 }, (Tensor)new[] { 0, 1 }, (Tensor)new[] { 2, 0 } },
            new object[] { new[] { 1, 1 }, (Tensor)new[] { 0, 1, 1, 0 }, (Tensor)new[] { 1, 1, 3, 2 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoPadsPositiveData))]
    public void TestFoldTwoPadsPositive(int[] shape, Tensor pads1, Tensor pads2)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(NN.Pad(a, pads1, PadMode.Constant, 0.0f), pads2, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoPads() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
