using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Tensorflow.Operations.Initializers;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestIntegralPromotion
{
    private readonly RunPassOptions passOptions;

    public UnitTestIntegralPromotion()
    {
        passOptions = new RunPassOptions(null, 3, Testing.GetDumpDirPath(this.GetType()));
    }

    public static IEnumerable<object[]> TestFoldNopBinaryNegativeData =>
        new[]
        {
            new object[] {DataTypes.Int32, DataTypes.Int64},
            new object[] {DataTypes.Int64, DataTypes.Int32},
        };

    [Theory]
    [MemberData(nameof(TestFoldNopBinaryNegativeData))]
    public void TestFoldNopBinaryNegative(DataType aType, DataType bType)
    {
        var caseOptions = passOptions.IndentDir($"{aType}_{bType}");
        // var a = new Var();
        // var b = new Var();
        // var Normal = new Dictionary<Var, IValue>();
        // Normal.Add(a, Random.Normal(aType, 0, 1, 0, new[]{2,2}).Evaluate());
        // Normal.Add(b, Random.Normal(bType, 0, 1, 0, new[]{2,2}).Evaluate());
        var a = Random.Normal(aType, 0, 1, 0, new[] {2, 2});
        var b = Random.Normal(bType, 0, 1, 0, new[] {2, 2});
        var rootPre = Math.Binary(BinaryOp.Add, a, b);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new IntegralPromotion(),
        }, caseOptions);
        // rootPre.InferenceType();
        Assert.NotEqual(rootPre, rootPost);
        // Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

}