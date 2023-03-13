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
public class UnitTestReluToClamp : TransformTestBase
{
    public static readonly TheoryData<Nncase.IR.NN.ActivationOp, int[]> ReluToClampPositiveData = new()
    {
        { new IR.NN.Relu(), new[] { 1, 2, 3, 4 } },
        { new IR.NN.Relu6(), new[] { 4, 3, 2, 1 } },
    };

    public static readonly TheoryData<Nncase.IR.NN.ActivationOp, int[]> ReluToClampNegativeData = new()
    {
        { new IR.NN.LeakyRelu(), new[] { 1, 2, 3, 4 } },
        { new IR.NN.LeakyRelu(), new[] { 4, 3, 2, 1 } },
    };

    [Theory]
    [MemberData(nameof(ReluToClampPositiveData))]
    public void TestPositive(Nncase.IR.NN.ActivationOp op, int[] shape)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var rootPre = new Call(op, input);

        var feedDict = new Dictionary<Var, IValue>()
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape).Evaluate() },
        };
        TestMatchedCore(
            rootPre,
            feedDict,
            new IRewriteRule[]
            {
               new ReluToClamp(),
               new Relu6ToClamp(),
            });
    }

    [Theory]
    [MemberData(nameof(ReluToClampNegativeData))]
    public void TestNegative(Nncase.IR.NN.ActivationOp op, int[] shape)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var rootPre = new Call(op, input);
        TestNotMatch(
            rootPre,
            new IRewriteRule[]
            {
               new ReluToClamp(),
               new Relu6ToClamp(),
            });
    }
}
