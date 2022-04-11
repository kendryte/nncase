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
            new[] { DataTypes.Int8, DataTypes.Int16, DataTypes.UInt32 },
            new[] { DataTypes.Int8, DataTypes.Int16, DataTypes.Float32 },
            new[] { DataTypes.UInt8, DataTypes.Float32, DataTypes.UInt64 },
            new[] { DataTypes.BFloat16, DataTypes.UInt32, DataTypes.Int64 },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoCastsPositiveData))]
    public void TestFoldTwoCastsPositive(DataType preType, DataType middleType, DataType PostType)
    {
        var caseOptions = passOptions.IndentDir($"{preType.GetDisplayName()}_{middleType.GetDisplayName()}_{PostType.GetDisplayName()}");
        var a = Random.Normal(preType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(Tensors.Cast(a, middleType), PostType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoCasts() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldTwoCastsNegativeData =>
        new[]
        {
            new[] { DataTypes.Int8, DataTypes.UInt8, DataTypes.UInt32 },
            new[] { DataTypes.UInt8, DataTypes.Int16, DataTypes.Int8 },
            new[] { DataTypes.UInt8, DataTypes.Float32, DataTypes.BFloat16 },
            new[] { DataTypes.BFloat16, DataTypes.UInt32, DataTypes.UInt8 },
        };

    [Theory]
    [MemberData(nameof(TestFoldTwoCastsNegativeData))]
    public void TestFoldTwoCastsNegative(DataType preType, DataType middleType, DataType PostType)
    {
        var caseOptions = passOptions.IndentDir($"{preType.GetDisplayName()}_{middleType.GetDisplayName()}_{PostType.GetDisplayName()}");
        var a = Random.Normal(preType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(Tensors.Cast(a, middleType), PostType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoCasts() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
    public static IEnumerable<object[]> TestFoldNopCastPositiveData =>
        new[]
        {
            new[] { DataTypes.Int8    ,DataTypes.Int8     },
            new[] { DataTypes.Int16   ,DataTypes.Int16    },
            new[] { DataTypes.Int32   ,DataTypes.Int32    },
            new[] { DataTypes.Int64   ,DataTypes.Int64    },
            new[] { DataTypes.UInt8   ,DataTypes.UInt8    },
            new[] { DataTypes.UInt16  ,DataTypes.UInt16   },
            new[] { DataTypes.UInt32  ,DataTypes.UInt32   },
            new[] { DataTypes.UInt64  ,DataTypes.UInt64   },
            new[] { DataTypes.BFloat16,DataTypes.BFloat16 },

        };

    [Theory]
    [MemberData(nameof(TestFoldNopCastPositiveData))]
    public void TestFoldNopCastPositive(DataType preType, DataType postType)
    {
        var caseOptions = passOptions.IndentDir($"{preType.GetDisplayName()}_{postType.GetDisplayName()}");
        var a = Random.Normal(preType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(a, postType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopCast() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldNopCastNegativeData =>
       new[]
       {
            new[] { DataTypes.Int8    ,DataTypes.UInt8     },
            new[] { DataTypes.Int16   ,DataTypes.Int8    },
            new[] { DataTypes.Int32   ,DataTypes.Int64    },
            new[] { DataTypes.Int64   ,DataTypes.Int32    },
            new[] { DataTypes.UInt8   ,DataTypes.Int8    },
            new[] { DataTypes.UInt16  ,DataTypes.BFloat16   },
            new[] { DataTypes.UInt32  ,DataTypes.UInt64   },
            new[] { DataTypes.UInt64  ,DataTypes.Int8   },
            new[] { DataTypes.BFloat16,DataTypes.Float32 },

       };

    [Theory]
    [MemberData(nameof(TestFoldNopCastNegativeData))]
    public void TestFoldNopCastNegative(DataType preType, DataType postType)
    {
        var caseOptions = passOptions.IndentDir($"{preType.GetDisplayName()}_{postType.GetDisplayName()}");
        var a = Random.Normal(preType, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Tensors.Cast(a, postType);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopCast() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
