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
using static Nncase.IR.F.NN;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestAddMarker : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestAddMarkerRelu()
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Relu(a);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new AddRangeOfAndMarkerToRelu() }, caseOptions);

        CompilerServices.DumpDotIR(rootPost, "", caseOptions.DumpDir, false);
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    [Fact]
    public void TestAddMarkerTargetConst()
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var b = Relu(a);
        var pre = IR.F.Math.RangeOfMarker(new[] { 1, 2, 3, 4 }, b);
        CompilerServices.InferenceType(pre);
        CompilerServices.DumpDotIR(pre, "", caseOptions.DumpDir, false);
    }

    [Fact]
    public void TestAddMarkerAttrConst()
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var b = Relu(a);
        var pre = IR.F.Math.RangeOfMarker(b, new[] { 1, 2, 3, 4 });
        CompilerServices.InferenceType(pre);
        CompilerServices.DumpDotIR(pre, "", caseOptions.DumpDir, false);
    }

    [Fact]
    public void TestAddMarkerAllConst()
    {
        var caseOptions = GetPassOptions();
        var pre = IR.F.Math.RangeOfMarker(new[] { 4, 5, 6, 7 }, new[] { 1, 2, 3, 4 });
        CompilerServices.InferenceType(pre);
        CompilerServices.DumpDotIR(pre, "", caseOptions.DumpDir, false);
    }

    [Fact]
    public void TestAddMarkerWithTuple()
    {
        var caseOptions = GetPassOptions();
        var a = new IR.Tuple((IR.Const)1 * (IR.Const)2);
        var b = new IR.Tuple(a, a, a, a);
        var c = new IR.Tuple(b, b, b, b);
        var d = new IR.Tuple(c, c, c, c);
        var e = new IR.Tuple(d, d, d, d);
        var pre = IR.F.Math.RangeOfMarker(new[] { 4, 5, 6, 7 }, e);
        CompilerServices.InferenceType(pre);
        CompilerServices.DumpDotIR(pre, "", caseOptions.DumpDir, false);
    }
}
