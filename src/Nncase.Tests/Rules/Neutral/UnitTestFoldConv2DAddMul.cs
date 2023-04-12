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
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldConv2DAddMul : TransformTestBase
{
    public static readonly TheoryData<int[], (int, int), (int, int)> FoldConv2DAddMulPositiveData = new()
    {
        { new[] { 1, 256, 56, 56 }, (1, 1), (0, 0) },
        { new[] { 1, 32, 64, 64 }, (1, 1), (0, 0) },
    };

    public static readonly TheoryData<int[], (int, int), (int, int)> FoldConv2DAddMulNegativeData = new()
    {
        { new[] { 1, 32, 56, 56 }, (3, 3), (1, 1) },
    };

    [Theory]
    [MemberData(nameof(FoldConv2DAddMulPositiveData))]
    public void TestPositive(int[] shape, (int KernelH, int KernelW) kernel, (int PadH, int PadW) pad)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr rootPre;
        {
            var v0 = input + input;
            var v1 = v0 * Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 5, new[] { 1, shape[1], 1, 1 }).Evaluate());
            var v2 = v1 + Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 6, new[] { 1, shape[1], 1, 1 }).Evaluate());
            var v3 = IR.F.NN.Conv2D(
                v2,
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, shape[1], kernel.KernelH, kernel.KernelW }).Evaluate().AsTensor(),
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { pad.PadH, pad.PadH },
                    { pad.PadW, pad.PadW },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f });
            rootPre = v3;
        }

        var feedDict = new Dictionary<Var, IValue>()
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape).Evaluate() },
        };
        TestMatchedCore(
            rootPre,
            feedDict,
            new IRewriteRule[]
            {
               new FoldConv2DAddMul(),
               new FoldConstCall(),
            });
    }

    [Fact]
    public void TestPositive2()
    {
        int[] shape = new[] { 1, 224, 224, 3 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr rootPre;
        {
            var v0 = IR.F.Tensors.Transpose(input, new[] { 0, 3, 1, 2 }); // f32[1,3,224,224]
            var v1 = IR.F.NN.Conv2D(
                          v0,
                          IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, 3, 7, 7 }).Evaluate().AsTensor(),
                          IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                          new[] { 2, 2 },
                          new[,]
                            {
                                { 3, 3 },
                                { 3, 3 },
                            },
                          new[] { 1, 1 },
                          PadMode.Constant,
                          1,
                          new[] { 0.0f, 6.0f }); // f32[1,64,112,112]

            var v2 = IR.F.NN.ReduceWindow2D(ReduceOp.Mean, v1, -3.4028235E+38F, new[] { 3, 3 }, new[] { 2, 2 }, new[,] { { 1, 1 }, { 1, 1 }, }, new[] { 1, 1 }, false, false); // f32[1,64,56,56]
            var v3 = IR.F.Math.Mul(v2, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 64, 1, 1 })); // f32[1,64,56,56]
            var v4 = IR.F.Math.Add(v3, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 64, 1, 1 })); // f32[1,64,56,56]
            var v5 = IR.F.NN.Relu(v4); // f32[1,64,56,56]
            var v6 = IR.F.NN.Conv2D(
                          v5,
                          IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 256, 64, 1, 1 }).Evaluate().AsTensor(),
                          IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 256 }).Evaluate().AsTensor(),
                          new[] { 1, 1 },
                          new[,]
                            {
                                { 0, 0 },
                                { 0, 0 },
                            },
                          new[] { 1, 1 },
                          PadMode.Constant,
                          1,
                          new[] { 0.0f, 6.0f }); // f32[1,256,56,56]
            rootPre = v6;
        }

        var feedDict = new Dictionary<Var, IValue>()
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape).Evaluate() },
        };
        var rootPost = TestMatchedCore(
            rootPre,
            feedDict,
            new IRewriteRule[]
            {
            new ReluToClamp(),
            new CombineClampAdd(),
            new CombineClampMul(),
            new FoldConv2DAddMul(),
            new FoldConstCall(),
            });

        Assert.True(rootPost is Call { Target: IR.NN.Conv2D } rootCall &&
          rootCall[IR.NN.Conv2D.Input] is Call { Target: IR.Math.Clamp });
    }

    [Theory]
    [MemberData(nameof(FoldConv2DAddMulNegativeData))]
    public void TestNegative(int[] shape, (int KernelH, int KernelW) kernel, (int PadH, int PadW) pad)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr rootPre;
        {
            var v0 = input + input;
            var v1 = v0 * new Var(new TensorType(DataTypes.Float32, new[] { 1, shape[1], 1, 1 }));
            var v2 = v1 + new Var(new TensorType(DataTypes.Float32, new[] { 1, shape[1], 1, 1 }));
            var v3 = IR.F.NN.Conv2D(
                v2,
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64, shape[1], kernel.KernelH, kernel.KernelW }).Evaluate().AsTensor(),
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 64 }).Evaluate().AsTensor(),
                new[] { 1, 1 },
                new[,]
                {
                    { pad.PadH, pad.PadH },
                    { pad.PadW, pad.PadW },
                },
                new[] { 1, 1 },
                PadMode.Constant,
                1,
                new[] { 0.0f, 6.0f });
            rootPre = v3;
        }

        TestNotMatch(
            rootPre,
            new IRewriteRule[]
            {
               new FoldConv2DAddMul(),
               new FoldConstCall(),
            });
    }
}
