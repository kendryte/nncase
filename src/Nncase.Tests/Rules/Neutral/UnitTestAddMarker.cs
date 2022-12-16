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

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
