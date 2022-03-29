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

public class UnitTestFoldClamp
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldClamp()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
    }

    public static IEnumerable<object[]> TestFoldNopClampPositiveData =>
        new[]
        {
            new object[] { float.NegativeInfinity, float.PositiveInfinity },
            new object[] { float.MinValue, float.MaxValue },
            // new object[] { double.NegativeInfinity, double.PositiveInfinity },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopClampPositiveData))]
    public void TestFoldNopCastPositive(float min, float max, int index)
    {
        var caseOptions = passOptions.IndentDir($"case_{index}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Math.Clamp(a, min, max);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopClamp() }, caseOptions);
    
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
