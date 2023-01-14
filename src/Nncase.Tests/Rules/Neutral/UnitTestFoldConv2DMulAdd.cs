// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldConv2DMulAdd : TestClassBase
{
    public static TheoryData<int[], (int, int), (int, int)> FoldConv2DMulAddPositiveData = new()
    {
        { new[] { 1, 256, 56, 56 }, (1, 1), (0, 0) },
        { new[] { 1, 32, 64, 64 }, (1, 1), (0, 0) },
        { new[] { 1, 32, 56, 56 }, (3, 3), (1, 1) },
    };

    [Theory]
    [MemberData(nameof(FoldConv2DMulAddPositiveData))]
    public void TestPositive(int[] shape, (int, int) kernel, (int, int) pad)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr rootPre;
        {
            var v0 = input + input;
            var v1 = v0 * Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 5, new[] { 1, shape[1], 1, 1 }).Evaluate());
            var v2 = v1 + Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 5, new[] { 1, shape[1], 1, 1 }).Evaluate());
            var v3 = IR.F.NN.Conv2D(
                v2,
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, shape[1], kernel.Item1, kernel.Item2 }).Evaluate().AsTensor(),
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(), new[] { 1, 1 },
                new[,]
                  {
                    { pad.Item1, pad.Item1 },
                    { pad.Item2, pad.Item2 },
                  }, new[] { 1, 1 }, PadMode.Constant, 1,
                new[] { 0.0f, 6.0f });
            rootPre = v3;
        }

        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
           new FoldConv2DMulAdd(),
           new FoldConstCall(),
        }, new());

        Dumpper.DumpIR(rootPost, "post");

        var feedDict = new Dictionary<Var, IValue>()
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape).Evaluate() },
        };
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, feedDict), CompilerServices.Evaluate(rootPost, feedDict));
    }

    [Theory]
    [MemberData(nameof(FoldConv2DMulAddPositiveData))]
    public void TestNegative(int[] shape, (int, int) kernel, (int, int) pad)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr rootPre;
        {
            var v0 = input + input;
            var v1 = v0 * IR.F.Random.Normal(DataTypes.Float32, 0, 1, 5, new[] { 1, shape[1], 1, 1 });
            var v2 = v1 + IR.F.Random.Normal(DataTypes.Float32, 0, 1, 5, new[] { 1, shape[1], 1, 1 });
            var v3 = IR.F.NN.Conv2D(
                v2,
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, shape[1], kernel.Item1, kernel.Item2 }).Evaluate().AsTensor(),
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(), new[] { 1, 1 },
                new[,]
                  {
                    { pad.Item1, pad.Item1 },
                    { pad.Item2, pad.Item2 },
                  }, new[] { 1, 1 }, PadMode.Constant, 1,
                new[] { 0.0f, 6.0f });
            rootPre = v3;
        }

        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
           new FoldConv2DMulAdd(),
           new FoldConstCall(),
        }, new());

        Assert.Equal(rootPre, rootPost);
    }
}
