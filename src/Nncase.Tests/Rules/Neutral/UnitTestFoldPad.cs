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

public class UnitTestFoldPad
{
    private readonly RunPassOptions passOptions;

    public UnitTestFoldPad()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
    }

    public static IEnumerable<object[]> TestFoldNopPadPositiveData =>
        new[]
        {
            new object[] { new[] { 1 }, (Tensor)new[,] { { 0, 0 } } },
            new object[] { new[] { 1, 1 }, (Tensor)new[,] { { 0, 0 }, { 0, 0 } } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopPadPositiveData))]
    public void TestFoldNopPadPositive(int[] shape, Tensor pads, int index)
    {
        var caseOptions = passOptions.IndentDir($"case_{index}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(a, pads, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopPad() }, passOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    public static IEnumerable<object[]> TestFoldTwoPadsPositiveData =>
        new[]
        {
            new object[] { new[] { 1 }, (Tensor)new[,] { { 0, 1 } }, (Tensor)new[,] { { 2, 0 } } },
            new object[] { new[] { 1, 1 }, (Tensor)new[,] { { 0, 1 }, { 1, 0 } }, (Tensor)new[,] { { 1, 3 }, { 1, 2 } } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldTwoPadsPositiveData))]
    public void TestFoldTwoPadsPositive(int[] shape, Tensor pads1, Tensor pads2, int index)
    {
        var caseOptions = passOptions.IndentDir($"case_{index}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NN.Pad(NN.Pad(a, pads1, PadMode.Constant, 0.0f), pads2, PadMode.Constant, 0.0f);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldTwoPads() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}

// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// using System.Runtime.CompilerServices;
// using System.Text;
// using System.Threading.Tasks;
// using Nncase.IR.F;
// using Nncase.Transform;
// using Nncase.Transform.Rules.Neutral;
// using Xunit;
// using Math = Nncase.IR.F.Math;
// using Random = Nncase.IR.F.Random;

// namespace Nncase.Tests.Rules.Neutral;

// public class UnitTestFoldClamp
// {
//     private readonly RunPassOptions passOptions;

//     public UnitTestFoldClamp()
//     {
//         string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "..", "tests_output");
//         dumpDir = Path.GetFullPath(dumpDir);
//         Directory.CreateDirectory(dumpDir);
//         passOptions = new RunPassOptions(null, 3, dumpDir);
//     }

//     private static string GetThisFilePath([CallerFilePath] string path = null)
//     {
//         return path;
//     }

//     public static int T = 1;

//     public static IEnumerable<object[]> TestFoldNopClampPositiveData
//     {
//         get
//         {
//             Console.WriteLine("GetFold");
//             return new[]
//             {
//                 new Tensor[] {float.NegativeInfinity, float.PositiveInfinity},
//                 new Tensor[] {float.MinValue, float.MaxValue},
//                 new Tensor[] {double.NegativeInfinity, double.PositiveInfinity},
//             };
//         }
//         set { }
//     }



//     // public static IEnumerable<object[]> TestFoldNopClampPositiveData =>
//     //     new[]
//     //     {
//     //         new Tensor[] { float.NegativeInfinity, float.PositiveInfinity },
//     //         new Tensor[] { float.MinValue, float.MaxValue },
//     //         new Tensor[] { double.NegativeInfinity, double.PositiveInfinity },
//     //     };

//     [Theory]
//     [MemberData(nameof(TestFoldNopClampPositiveData))]
//     public void TestFoldNopCastPositive(Tensor min, Tensor max)
//     {
//         var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
//         var rootPre = Math.Clamp(a, min, max);
//         var rootPost = CompilerServices.Rewrite(rootPre, new[] { new FoldNopClamp() }, passOptions);

//         Assert.NotEqual(rootPre, rootPost);
//         Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
//     }
// }