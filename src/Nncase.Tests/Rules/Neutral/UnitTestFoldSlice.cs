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

public class UnitTestFoldSlice
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldSlice()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
    }

    public static IEnumerable<object[]> TestFoldNopSlicePositiveData =>
        new[]
        {
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0 }, new[] { 6 }, new[] { 2 }, new[] { 1 } },
            new object[] { new[] { 2, 4, 6, 8 }, new[] { 0, 0 }, new[] { 4, 6 }, new[] { 1, 2 }, new[] { 1, 1 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopSlicePositiveData))]
    public void TestFoldNopSlicePositive(int[] shape, int[] begins, int[] ends, int[] axes, int[] strides, int index)
    {
        var caseOptions = passOptions.IndentDir($"case_{index}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Slice(a, begins, ends, axes, strides);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopSlice() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    // TODO: Add tests for FoldTwoSlices
}
