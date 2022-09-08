﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestAddToConv2D: TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestElementwiseAdd()
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = a + b;
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new AddToConv2D() }, caseOptions);

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    [Fact]
    public void TestNegElementwiseAdd()
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 1, 8, 8 });
        var rootPre = a + b;
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new AddToConv2D() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }
}
