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
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestFoldCast
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldCast()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
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
        var caseOptions = passOptions.IndentDir($"{preAndPostType.GetDisplayName()}_{middleType.GetDisplayName()}");
        var a = Random.Normal(preAndPostType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(Tensors.Cast(a, middleType), preAndPostType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoCasts() }, caseOptions);

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
        var caseOptions = passOptions.IndentDir($"{dataType.GetDisplayName()}");
        var a = Random.Normal(dataType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(a, dataType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopCast() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
