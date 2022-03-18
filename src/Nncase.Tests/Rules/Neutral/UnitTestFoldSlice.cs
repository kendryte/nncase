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

public class UnitTestFoldSlice
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldSlice()
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

    public static IEnumerable<object[]> TestFoldNopSlicePositiveData =>
        new[]
        {
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0 }, new[] { 6 }, new[] { 2 }, new[] { 1 } },
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0, 0 }, new[] { 4, 6 }, new[] { 1, 2 }, new[] { 1, 1 } },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopSlicePositiveData))]
    public void TestFoldNopSlicePositive(int[] shape, int[] begins, int[] ends, int[] axes, int[] strides)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Slice(a, begins, ends, axes, strides);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopSlice() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    // TODO: Add tests for FoldTwoSlices
}
