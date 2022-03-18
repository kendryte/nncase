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

public class UnitTestFoldClamp
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldClamp()
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

    public static IEnumerable<object[]> TestFoldNopClampPositiveData =>
        new[]
        {
            new Tensor[] { float.NegativeInfinity, float.PositiveInfinity },
            new Tensor[] { float.MinValue, float.MaxValue },
            new Tensor[] { double.NegativeInfinity, double.PositiveInfinity },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopClampPositiveData))]
    public void TestFoldNopCastPositive(Tensor min, Tensor max)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Math.Clamp(a, min, max);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopClamp() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
