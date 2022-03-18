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

namespace Nncase.Tests.Rules.Neutral;

public class UnitTestFoldCast
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldCast()
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

    public static IEnumerable<object[]> TestFoldTwoCastsPositiveData =>
        new[]
        {
            new[] { DataTypes.Int8, DataTypes.Int16 },
            new[] { DataTypes.Int8, DataTypes.Int32 },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoCastsPositiveData))]
    public void TestFoldTwoCastsPositive(DataType preAndPostType, DataType middleType)
    {
        var a = Random.Normal(preAndPostType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(Tensors.Cast(a, middleType), preAndPostType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoCasts() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldNopCastPositiveData =>
        new[]
        {
            new[] { DataTypes.Int8 },
            new[] { DataTypes.Int16 },
            new[] { DataTypes.Int32 },
        };

    [Theory]
    [MemberData(nameof(TestFoldNopCastPositiveData))]
    public void TestFoldNopCastPositive(DataType dataType)
    {
        var a = Random.Normal(dataType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(a, dataType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopCast() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
