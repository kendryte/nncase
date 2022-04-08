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

public class UnitTestFusePadConv2D
{
    private readonly RunPassOptions passOptions;

    public UnitTestFusePadConv2D()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
    }

    public static IEnumerable<object[]> TestFusePadConv2DPositiveData =>
        new[]
        {
            new object[] { new[] { 1, 1, 2, 2 }, new[,] { { 0, 0 },{ 0, 0 },{ 3, 3 },{ 4, 4 } }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 3, 1, 1, 1 } },
            new object[] { new[] { 1, 3, 4, 2 }, new[,] { { 0, 0 },{ 0, 0 },{ 5, 0 },{ 1, 3 } }, new[,] { { 0, 2 }, { 3, 2 } }, new[] { 1, 3, 2, 2 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFusePadConv2DPositiveData))]
    public void TestFusePadConv2DPositive(int[] shape, int[,] pads1, int[,] pads2, int[] wShape, int index)
    {
        //TODO 当fuse之后的pad数超过了input shape的尺寸 不能fuse
        var caseOptions = passOptions.IndentDir($"case_{index}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var w = Random.Normal(DataTypes.Float32, 0, 1, 0, wShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { wShape[0] });
        var rootPre = NN.Conv2D(NN.Pad(a, pads1, PadMode.Constant, 0f), w, b, new[] { 1, 1 }, pads2, new[] { 1, 1 }, PadMode.Constant, 1);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FusePadConv2d(),
            new FoldConstCall(),
            new FoldNopPad(),
            
        }, caseOptions);
        

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
    
    public static IEnumerable<object[]> TestFusePadConv2DNegativeData =>
        new[]
        {
            new object[] { new[] { 1, 1, 2, 2 }, new[,] { { 0, 0 },{ 1, 1 },{ 0, 0 },{ 0, 0 } }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 3, 3, 1, 1 } },
            new object[] { new[] { 1, 3, 4, 2 }, new[,] { { 1, 1 },{ 0, 0 },{ 0, 0 },{ 0, 0 } }, new[,] { { 0, 2 }, { 3, 2 } }, new[] { 3, 3, 2, 2 } },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFusePadConv2DNegativeData))]
    public void TestFusePadConv2DNegative(int[] shape, int[,] pads1, int[,] pads2, int[] wShape, int index)
    {
        var caseOptions = passOptions.IndentDir($"case_{index}");
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var w = Random.Normal(DataTypes.Float32, 0, 1, 0, wShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { wShape[0] });
        var rootPre = NN.Conv2D(NN.Pad(a, pads1, PadMode.Constant, 0f), w, b, new[] { 1, 1 }, pads2, new[] { 1, 1 }, PadMode.Constant, 1);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FusePadConv2d(),
            
        }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
