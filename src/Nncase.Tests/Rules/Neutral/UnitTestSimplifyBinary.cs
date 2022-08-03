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

public class UnitTestSimplifyBinary: TestFixture.UnitTestFixtrue
{
    public static IEnumerable<object[]> TestReassociateMulPositiveData =>
        new[]
        {
            new object[] {new[] {3}},
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestReassociateMulPositiveData))]
    public void TestReassociateMulPositive(int[] aShape, int index)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var b = new Var();
        var c = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        Normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        Normal.Add(c, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = a * b * c; //Math.Binary(binaryOp, Math.Binary(binaryOp, a, bValue), bValue);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new ReassociateMul(),
        }, caseOptions);
        // rootPre.InferenceType();
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    public static IEnumerable<object[]> TestReassociateDivPositiveData =>
        new[]
        {
            new object[] {new[] {3}},
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestReassociateDivPositiveData))]
    public void TestReassociateDivPositive(int[] aShape, int index)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var b = new Var();
        var c = Random.Normal(DataTypes.Float32, 0, 1, 0, aShape); // Can't get Var's datatype. Pattern will not pass 
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        Normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = (a * b) / c;
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new ReassociateDiv(),
        }, caseOptions);
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }


    public static IEnumerable<object[]> TestXDivXPositiveData =>
        new[]
        {
            new object[] {new[] {3}},
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestXDivXPositiveData))]
    public void TestXDivXPositive(int[] aShape, int index)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = a / a;
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new XDivX(),
        }, caseOptions);
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    public static IEnumerable<object[]> TestCommutateMulPositiveData =>
        new[]
        {
            new object[] {new[] {3}},
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestCommutateMulPositiveData))]
    public void TestCommutateMulPositive(int[] aShape, int index)
    {
        var caseOptions = GetPassOptions();
        caseOptions = caseOptions.SetRewriteOnce(true);
        var a = new Var();
        var b = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        Normal.Add(b, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        var rootPre = a * b;
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CommutateMul(),
        }, caseOptions);
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }
}